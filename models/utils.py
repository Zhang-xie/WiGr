import torch


def custom_stack(x,time_dim=2,time_size=1800):
    out=[]
    print("00")
    if isinstance(x,list):
        slc = [slice(None)] * len(x[0].shape)
        slc[time_dim] = slice(0, time_size)
        r_slc=[1]*len(x[0].shape)
        print(slc)
        print(r_slc)
        for i in x:
            if i.shape[time_dim]<time_size:
                r_slc[time_dim]=1+time_size//i.shape[time_dim]
                i=i.repeat(*r_slc)
            if i.shape[time_dim]>time_size:
                out.append(i[slc])
    return torch.stack(out)


if __name__ == '__main__':
    a = [[1,2],[2,3],[3,4]]
    x1 = torch.Tensor(a)
    x2 = torch.Tensor(a)
    x = [x1,x2]

    print(custom_stack(x,time_dim=1,time_size=5))




