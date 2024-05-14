import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Normalization of Causality information')

    parser.add_argument('--list_type', nargs='+', type=int, default=[0, 0, 0, 0, 0, 2, 0, 0, 2, 2],
                        help='variable type list [0 means CV, integer more than 1 means DV]')

    parser.add_argument('--list_type', nargs='+', type=int, default=[0, 0, 0, 0, 0, 2, 0, 0, 2, 2],
                        help='variable type list [0 means CV, integer more than 1 means DV]')


    args = parser.parse_args()

    print(args.list_type)

# self.lag = 5
#         self.num_series = 3
#         self.batch_size = 2
#         self.list_type = [np.inf, np.inf, 2]
#
#         self.embed_type = 'val&pos'
#         self.len_embed_ch = 5
#         self.len_embed_val = 5
#
#         self.num_layers = {}
#
#         self.size_projection = 10
#         self.num_layers['cnn'] = 3
#
#         self.num_layers['mlp'] = [100]
#
#         self.output_attn = False
#         self.num_heads = 5
#
#         self.activation = 'relu'
#         self.dropout = 0.5