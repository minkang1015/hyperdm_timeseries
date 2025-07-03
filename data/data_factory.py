from data.data_loader import Dataset_Custom, Dataset_SNP, Dataset_DOW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

data_dict = {
    'SNP': Dataset_SNP,
    'DOW': Dataset_DOW,
    'custom': Dataset_Custom,
}

def data_provider(args, flag, shuffle_flag=True):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    else:
        shuffle_flag = shuffle_flag
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    data_set = Data(
        root_path=args.root_path,
        # data_path=args.`data_path`,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        timeenc=timeenc,
        freq=freq,
        scale=args.inverse,
        train_end=args.train_end,
        val_end=args.valid_end,
    )
    # print(flag, len(data_set))

    data_loader = DataLoader(
                data_set,
                batch_size=batch_size,
                shuffle= shuffle_flag,
                #num_workers=args.num_workers,   
                drop_last=drop_last)
    
    # print(flag, len(data_loader))
    print(f"{flag} length: {len(data_set)}, number of batches: {len(data_loader)}")

    return data_set, data_loader