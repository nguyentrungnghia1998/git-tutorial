from src.data_helper import *
from models.SharedNet import SharedNet
import argparse		# Thư viện giúp tạo định nghĩa command line trong terminal

from src.trainer import Trainer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default='input/test_subsample.vcf.gz')
    parser.add_argument('--index-path', type=str, default='input/test.list')
    parser.add_argument('--label-path', type=str, default='input/DGV4VN_1015.HISAT_result.csv')
    parser.add_argument('--model-name', type=str, default='model.pt')
    parser.add_argument('--output-path', type=str, default='output')
    parser.add_argument('-l', '--loss', type=str, default='cross_entropy')
    parser.add_argument('-o', '--optimizer', type=str, default='adam')
    parser.add_argument('-e', '--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-n', '--n-repeats', type=int, default=1)
    parser.add_argument('-p', '--print-every', type=int, default=5)
    parser.add_argument('-s', '--save-every', type=int, default=10)
    parser.add_argument('-d', '--save-dir', type=str, default='./trainned_models')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--sample', type=int, default=1000)
    parser.add_argument('-m', '--transform', action='store_true', default=True)
    
    args = parser.parse_args() 
    return args


def main():
    args = parse_args()
    if args.transform:
        ''' read cvf file and convert to csv format '''
        df = load_vcf_file(args.data_path, 
                             index_path=args.index_path, 
                             nrows=args.sample,
                             saved=True)
    else: 
        df = pd.read_csv(args.data_path.replace('.vcf.gz', '.csv'), index_col=0)
        
    label_df_file_path = args.label_path
    label_df = pd.read_csv(label_df_file_path)
    label_df['Unnamed: 0']=label_df['Unnamed: 0'].apply(convert_index)
    label_df.set_index('Unnamed: 0',inplace=True)


    columns = label_df.columns
    index_df=df.index
    index_label_df=label_df.index
    index_all=[]
    for i in index_df:
        if i[:-2] in index_label_df:
            index_all.append(i)

    one_hot_encoder = {}

    for i in range(0, len(columns), 2):
        col_1 = label_df[columns[i]].values
        col_2 = label_df[columns[i+1]].values
        combined_col = sorted(set(np.concatenate((col_1, col_2), axis=0)))
        n_values = len(combined_col) + 1
        one_hot_vector = np.eye(n_values)[range(n_values)]
        for j, x in enumerate(combined_col):
            one_hot_encoder[(columns[i], x)] = one_hot_vector[j]
            one_hot_encoder[(columns[i+1], x)] = one_hot_vector[j]
        
    label_df_one_hot = label_df.copy()
    for col in columns:
        label_df_one_hot[col] = label_df_one_hot[col].apply(lambda x: one_hot_encoder[(col, x)])
    
    ''' One hot encoding '''
    allele_labels_1 = columns[::2]
    allele_labels_2 = columns[1::2]
    df_allele_labels_1 = label_df_one_hot[allele_labels_1]
    df_allele_labels_2 = label_df_one_hot[allele_labels_2]
    df_allele_labels_1.rename(index=lambda x: x + '_1', inplace=True)
    df_allele_labels_1.rename(columns=lambda x: x.replace('_1', ''), inplace=True)
    df_allele_labels_2.rename(index=lambda x: x + '_2', inplace=True)
    df_allele_labels_2.rename(columns=lambda x: x.replace('_2', ''), inplace=True)
    df_allele_labels = pd.concat([df_allele_labels_1, df_allele_labels_2], axis=0)
    
    columns = df_allele_labels.columns
    
    ''' concat of all columns is label_df_one_hot '''

    df_allele_labels['label'] = df_allele_labels.apply(
        lambda x: np.concatenate(x.values), axis=1)
    
    dataset = {
        'data': [],
        'label': []
    }
    for i in index_all:
        data_row = df.loc[i].values
        label = df_allele_labels['label'].loc[i]
        dataset['data'].append(data_row)
        dataset['label'].append(label)
    
    trainset, testset = split_dataset(dataset, 0.9, shuffle=True)
    
    input_size = len(dataset['data'][0])
    outputs_size = [['HLA_' + col, len(df_allele_labels[col].iloc[0])] for col in columns]
    print('input_size: {}, output_size: {}'.format(input_size, outputs_size))
    
    trainer = Trainer(
        model=SharedNet(input_size, outputs_size),
        train_loader=trainset,
        test_loader=testset,
        loss=args.loss,
        optimizer=args.optimizer,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        n_repeats=args.n_repeats,
        print_every=args.print_every,
        save_every=args.save_every,
        save_dir=args.save_dir,
        verbose=args.verbose
        )
    trainer.train()
    
if __name__ == "__main__":
    main()
