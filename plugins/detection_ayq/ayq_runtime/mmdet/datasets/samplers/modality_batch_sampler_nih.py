# mmdet/datasets/samplers/modality_batch_sampler.py

import random
from collections import defaultdict
from torch.utils.data import BatchSampler
from mmdet.registry import DATA_SAMPLERS
from pycocotools.coco import COCO


@DATA_SAMPLERS.register_module()
class ModalityBatchSamplerNIH(BatchSampler):
    def __init__(self,
                 sampler,       # 내부 sampler (DefaultSampler 등)
                 batch_size,    # 배치 크기
                 drop_last,     # 마지막 남은 배치 드롭 여부
                 dataset=None,  # build_dataloader 에서 전달됨
                 **kwargs):

        # 1) dataset 확보
        if dataset is not None:
            self.dataset = dataset
        elif hasattr(sampler, 'dataset'):
            self.dataset = sampler.dataset
        else:
            raise ValueError('ModalityBatchSampler requires a dataset!')

        # 2) mmengine BaseDataset 에는 full_init() → data_infos 를 채워줌
        if hasattr(self.dataset, 'full_init'):
            self.dataset.full_init()

        super().__init__(sampler, batch_size, drop_last)

        # -- COCO ann_file 로드 --
        ann_path = getattr(self.dataset, 'ann_file', None)
        if ann_path is None:
            raise ValueError('Your dataset must have an `ann_file` attribute')
        coco = COCO(ann_path)

        # -- dataset.data_list / data_infos 확보, 비어 있으면 직접 load_annotations() 호출 --
        data_list = getattr(self.dataset, 'data_list', None)
        data_infos = getattr(self.dataset, 'data_infos', None)
        if not data_list and not data_infos:
            # mmengine.Dataset 스타일
            if hasattr(self.dataset, 'load_data_list'):
                data_list = self.dataset.load_data_list()
                self.dataset.data_list = data_list
            # mmdet.CustomDataset 스타일
            elif hasattr(self.dataset, 'load_annotations'):
                data_list = self.dataset.load_annotations(ann_path)
                self.dataset.data_list = data_list
            else:
                raise ValueError('Cannot populate data_list/data_infos from dataset')

        # 이제 data_list 에 채워진 리스트를 씁니다.
        infos = data_list # self.dataset.data_list if data_list else self.dataset.data_infos

        # -- modality 별로 인덱스를 모아 둘 dict 생성 --
        self.indices_per_mod = defaultdict(list)
        for idx, info in enumerate(infos):
            img_id = info.get('img_id')
            if img_id is None:
                continue
            ann_ids = coco.getAnnIds(imgIds=[img_id], iscrowd=None)
            anns = coco.loadAnns(ann_ids)
            if not anns:
                continue
            cls_name = coco.loadCats([anns[0]['category_id']])[0]['name']
            # modality = cls_name.split()[-1]
            modality = cls_name.split('in ')[0]
            self.indices_per_mod[modality].append(idx)

        self.modalities = list(self.indices_per_mod.keys())
        if not self.modalities:
            raise ValueError('No modalities found; check your ann_file and metainfo')

    def __iter__(self):
        pools = {m: random.sample(idxs, len(idxs))
                 for m, idxs in self.indices_per_mod.items()}
        pointers = {m: 0 for m in self.modalities}
        max_rounds = max(len(idxs) for idxs in pools.values())

        for _ in  range(max_rounds):
            chosen = random.sample(self.modalities, self.batch_size)
            batch = []
            for m in chosen:
                if pointers[m] < len(pools[m]):
                    batch.append(pools[m][pointers[m]])
                    pointers[m] += 1
                    # MRI 같은 경우 최대 500개 정도인데, 
                    # 500개를 다 뽑아내면 다음에 뽑을게 없으므로
                    # 다시 처음으로 돌아가서 뽑아야 함
                else:
                    pointers[m] = 0
                    batch.append(pools[m][pointers[m]])
                    pointers[m] += 1
            # batch_size 보다 적은 경우 drop_last
            # 옵션이 True 이면 continue
            # False 이면 마지막 남은 배치도 yield
            if len(batch) < self.batch_size and self.drop_last:
                continue
            if batch:
                #print("Chosen modalities:", chosen)
                #print("pointers", pointers)
                #print("Batch indices:", batch)
                #print("Dataset length:", len(self.dataset))
                yield batch

    def __len__(self):
        return max(len(idxs) for idxs in self.indices_per_mod.values())
