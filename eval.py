import torch
from tqdm import tqdm
import utils
  

def eval_psnr(loader, model, eval_type=None, eval_bsize=None,
              verbose=False):
    model.eval()
    
    metric_fn = utils.calc_psnr

    val_res1 = utils.Averager()
    val_res2 = utils.Averager()
    val_res3 = utils.Averager()
    val_res0 = utils.Averager()


    pbar = tqdm(loader, leave=False, desc='val')
    for batch in pbar:
        for k, v in batch.items():
            batch[k] = v.cuda()

        if eval_bsize is None:
            with torch.no_grad():
                imgs = [batch['0'], batch['1'], batch['2'],batch['3']]
                
                pred_2 = model(imgs,[0,1,3])[0][2]
                pred_1 = model(imgs,[0,2,3])[0][1]
                pred_0 = model(imgs,[1,2,3])[0][0]
                pred_3 = model(imgs,[0,1,2])[0][3]
                
        res0 = metric_fn(pred_0, imgs[0])
        res1 = metric_fn(pred_1, imgs[1])
        res2 = metric_fn(pred_2, imgs[2])
        res3 = metric_fn(pred_3, imgs[3])
   
        val_res0.add(res0.item(), batch['1'].shape[0])
        val_res1.add(res1.item(), batch['1'].shape[0])
        val_res2.add(res2.item(), batch['1'].shape[0])
        val_res3.add(res3.item(), batch['1'].shape[0])


        if verbose:
            pbar.set_description('val {:.4f}'.format(val_res1.item()))

    return val_res0.item(),val_res1.item(),val_res2.item(),val_res3.item()

