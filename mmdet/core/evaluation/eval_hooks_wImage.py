import os.path as osp

import numpy as np
import mmcv

from mmdet.core import EvalHook
from mmdet.core.evaluation import eval_map
from mmdet.core.visualization import imshow_gt_det_bboxes
from mmdet.datasets import build_dataset, get_loading_pipeline

class EvalHookWImage(EvalHook):
    def evaluate(self, runner, results):
        runner.mode='val'
        eval_res = self.dataloader.dataset.evaluate(
            results, logger=runner.logger, **self.eval_kwargs)
        # add image res to tensorboard starts
        imgs = get_imgs(self.dataloader.dataset, results)
        runner.log_buffer.output['imgs'] = imgs
        # add image res to tensorboard ends
        for name, val in eval_res.items():
            runner.log_buffer.output[name] = val
        runner.log_buffer.ready = True
        if self.save_best is not None:
            if self.key_indicator == 'auto':
                # infer from eval_results
                self._init_rule(self.rule, list(eval_res.keys())[0])
            return eval_res[self.key_indicator]
        else:
            return None

class DistEvalHookWImage(EvalHookWImage):
    """Distributed evaluation hook.

    Notes:
        If new arguments are added, tools/test.py may be effected.

    Attributes:
        dataloader (DataLoader): A PyTorch dataloader.
        start (int, optional): Evaluation starting epoch. It enables evaluation
            before the training starts if ``start`` <= the resuming epoch.
            If None, whether to evaluate is merely decided by ``interval``.
            Default: None.
        interval (int): Evaluation interval (by epochs). Default: 1.
        tmpdir (str | None): Temporary directory to save the results of all
            processes. Default: None.
        gpu_collect (bool): Whether to use gpu or cpu to collect results.
            Default: False.
        save_best (str, optional): If a metric is specified, it would measure
            the best checkpoint during evaluation. The information about best
            checkpoint would be save in best.json.
            Options are the evaluation metrics to the test dataset. e.g.,
            ``bbox_mAP``, ``segm_mAP`` for bbox detection and instance
            segmentation. ``AR@100`` for proposal recall. If ``save_best`` is
            ``auto``, the first key will be used. The interval of
            ``CheckpointHook`` should device EvalHook. Default: None.
        rule (str | None): Comparison rule for best score. If set to None,
            it will infer a reasonable rule. Default: 'None'.
        **eval_kwargs: Evaluation arguments fed into the evaluate function of
            the dataset.
    """

    def __init__(self,
                 dataloader,
                 start=None,
                 interval=1,
                 tmpdir=None,
                 gpu_collect=False,
                 save_best=None,
                 rule=None,
                 **eval_kwargs):
        super().__init__(
            dataloader,
            start=start,
            interval=interval,
            save_best=save_best,
            rule=rule,
            **eval_kwargs)
        self.tmpdir = tmpdir
        self.gpu_collect = gpu_collect

    def after_train_epoch(self, runner):
        if not self.evaluation_flag(runner):
            return

        from mmdet.apis import multi_gpu_test
        tmpdir = self.tmpdir
        if tmpdir is None:
            tmpdir = osp.join(runner.work_dir, '.eval_hook')
        results = multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=tmpdir,
            gpu_collect=self.gpu_collect)
        if runner.rank == 0:
            print('\n')
            key_score = self.evaluate(runner, results)
            if self.save_best:
                best_score = runner.meta['hook_msgs'].get(
                    'best_score', self.init_value_map[self.rule])
                if self.compare_func(key_score, best_score):
                    best_score = key_score
                    runner.meta['hook_msgs']['best_score'] = best_score
                    last_ckpt = runner.meta['hook_msgs']['last_ckpt']
                    runner.meta['hook_msgs']['best_ckpt'] = last_ckpt
                    mmcv.symlink(
                        last_ckpt,
                        osp.join(runner.work_dir,
                                 f'best_{self.key_indicator}.pth'))
                    self.logger.info(
                        f'Now best checkpoint is {last_ckpt}.'
                        f'Best {self.key_indicator} is {best_score:0.4f}')

def get_imgs(dataset, results):
    show_nums = 20
    cnt = 0
    assert len(dataset) == len(results)
    imgs = []
    for idx in range(0, len(dataset), 10):
        img = dataset.prepare_test_img(idx)['img']
        bbox = results[idx][0]
        print(bbox)
        imgs.append((img, bbox))
        cnt += 1
        if cnt>=show_nums:
            break
    return imgs