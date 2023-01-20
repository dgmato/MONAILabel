# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import copy
import logging
import time
from typing import Callable, Sequence

import torch
from tqdm import tqdm

from monailabel.interfaces.tasks.infer_v2 import InferTask, InferType
from monailabel.interfaces.utils.transform import run_transforms
from monailabel.tasks.infer.basic_infer import BasicInferTask
from monailabel.transform.post import Restored
from monailabel.transform.writer import Writer

logger = logging.getLogger(__name__)


class InferDentalPipeline(BasicInferTask):
    def __init__(
        self,
        task_loc_teeth: InferTask,
        task_seg_teeth: InferTask,
        type=InferType.SEGMENTATION,
        description="Combines two stages for teeth segmentation",
        **kwargs,
    ):
        self.task_loc_teeth = task_loc_teeth
        self.task_seg_teeth = task_seg_teeth

        super().__init__(
            path=None,
            network=None,
            type=type,
            labels=task_seg_teeth.labels,
            dimension=task_seg_teeth.dimension,
            description=description,
            **kwargs,
        )

    def pre_transforms(self, data=None) -> Sequence[Callable]:
        return []

    def post_transforms(self, data=None) -> Sequence[Callable]:
        return []

    def is_valid(self) -> bool:
        return True

    def _latencies(self, r, e=None):
        if not e:
            e = {"pre": 0, "infer": 0, "invert": 0, "post": 0, "write": 0, "total": 0}

        for key in e:
            e[key] = e[key] + r.get("latencies", {}).get(key, 0)
        return e

    def locate_teeth(self, request):
        req = copy.deepcopy(request)
        req.update({"pipeline_mode": True})

        d, r = self.task_loc_teeth(req)
        return d, r, self._latencies(r)

    def segment_teeth(self, request, image, label):
        req = copy.deepcopy(request)
        req.update({"image": image, "label": label, "pipeline_mode": True})

        d, r = self.task_seg_teeth(req)       
        l = self._latencies(r)
        
        image = d["image"]
        image_cached = image
        
        result_mask = d["pred"]
        
        return d, r, self._latencies(r)

    def __call__(self, request):
        start = time.time()
        request.update({"image_path": request.get("image")})

        # Run first stage
        d1, r1, l1 = self.locate_teeth(request)
        image = d1["image"]
        label = d1["pred"]

        # Run second stage
        d2, r2, l2 = self.segment_teeth(request, image, label)
        result_mask = d2["pred"]

        # Finalize the mask/result
        data = copy.deepcopy(request)
        data.update({"pred": result_mask, "image": image})
        data = run_transforms(data, [Restored(keys="pred", ref_image="image")], log_prefix="POST(P)", use_compose=False)

        begin = time.time()
        result_file, _ = Writer(label="pred")(data)
        latency_write = round(time.time() - begin, 2)

        total_latency = round(time.time() - start, 2)
        result_json = {
            "label_names": self.task_seg_teeth.labels,
            "latencies": {
                "locate_teeth": l1,
                "segment_teeth": l2,
                "write": latency_write,
                "total": total_latency,
            },
        }

        logger.info(f"Result Mask (aggregated/pre-restore): {result_mask.shape}; total_latency: {total_latency}")
        return result_file, result_json
