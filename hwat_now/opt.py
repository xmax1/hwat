





def loss_table(
    *, 
    model  = None, 
    param  = None, 
    metric = []
):
    metric_fn = {
        'mse': lambda x: x,
        'l1_norm': lambda x, y: np.mean(np.abs(x - y))
    }
    for m in metric:
        metric_fn[m](param)
