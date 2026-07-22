import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from config import Config
from feature_engineering.data_engineering import data_engineer_benchmark, span_data_2d, span_data_3d
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
import pickle
import dgl
from scipy.io import loadmat
import yaml
from reproducibility import seed_everything

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)
# sys.path.append("..")


def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument("--method", default=str)  # specify which method to use
    parser.add_argument("--dataset", default=None)
    cli_args = vars(parser.parse_args())
    method = cli_args['method']  # dict
    dataset = cli_args['dataset']

    # if method in ['']:
    #     yaml_file = "config/base_cfg.yaml"
    if method in ['mcnn']:
        yaml_file = "config/mcnn_cfg.yaml"
    elif method in ['stan']:
        yaml_file = "config/stan_cfg.yaml"
    elif method in ['stan_2d']:
        yaml_file = "config/stan_2d_cfg.yaml"
    elif method in ['stagn']:
        yaml_file = "config/stagn_cfg.yaml"
    elif method in ['gtan']:
        yaml_file = "config/gtan_aml_cfg.yaml" if dataset == "aml" else "config/gtan_cfg.yaml"
    elif method in ['rgtan']:
        if dataset == "aml":
            yaml_file = "config/rgtan_aml_cfg.yaml"
        elif dataset == "amlsim":
            yaml_file = "config/rgtan_amlsim_cfg.yaml"
        else:
            yaml_file = "config/rgtan_cfg.yaml"
    elif method in ['rgtan_ca1']:
        if dataset in ("aml", "amlsim"):
            yaml_file = f"config/rgtan_{dataset}_ca1.yaml"
        else:
            raise ValueError("rgtan_ca1 supports only --dataset aml or amlsim")
    elif method in ['rgtan_ca1_ca3']:
        if dataset in ("aml", "amlsim"):
            yaml_file = f"config/rgtan_{dataset}_ca1_ca3.yaml"
        else:
            raise ValueError("rgtan_ca1_ca3 supports only --dataset aml or amlsim")
    elif method in ['rgtan_mpfc']:
        if dataset in ("aml", "amlsim"):
            yaml_file = f"config/rgtan_{dataset}_mpfc.yaml"
        else:
            raise ValueError("rgtan_mpfc supports only --dataset aml or amlsim")
    elif method in ['rgtan_mpfc_vta']:
        if dataset in ("aml", "amlsim"):
            yaml_file = f"config/rgtan_{dataset}_mpfc_vta.yaml"
        else:
            raise ValueError("rgtan_mpfc_vta supports only --dataset aml or amlsim")
    elif method in ['hogrl']:
        yaml_file = "config/hogrl_cfg.yaml"
        
    else:
        raise NotImplementedError("Unsupported method.")

    # config = Config().get_config()
    with open(yaml_file) as file:
        args = yaml.safe_load(file)
    args['method'] = method
    if dataset is not None:
        args['dataset'] = dataset
    return args


def base_load_data(args: dict):
    # load S-FFSD dataset for base models
    data_path = "data/S-FFSD.csv"
    feat_df = pd.read_csv(data_path)
    train_size = 1 - args['test_size']
    method = args['method']
    # for ICONIP16 & AAAI20
    if args['method'] == 'stan':
        if os.path.exists("data/tel_3d.npy"):
            return
        features, labels = span_data_3d(feat_df)
    else:
        if os.path.exists("data/tel_2d.npy"):
            return
        features, labels = span_data_2d(feat_df)
    num_trans = len(feat_df)
    trf, tef, trl, tel = train_test_split(
        features, labels, train_size=train_size, stratify=labels, shuffle=True)
    trf_file, tef_file, trl_file, tel_file = args['trainfeature'], args[
        'testfeature'], args['trainlabel'], args['testlabel']
    np.save(trf_file, trf)
    np.save(tef_file, tef)
    np.save(trl_file, trl)
    np.save(tel_file, tel)
    return


def main(args):
    if "seed" in args:
        args["reproducibility"] = seed_everything(args["seed"])
    if args['method'] == 'mcnn':
        from methods.mcnn.mcnn_main import mcnn_main
        base_load_data(args)
        mcnn_main(
            args['trainfeature'],
            args['trainlabel'],
            args['testfeature'],
            args['testlabel'],
            epochs=args['epochs'],
            batch_size=args['batch_size'],
            lr=args['lr'],
            device=args['device']
        )
    elif args['method'] == 'stan_2d':
        from methods.stan.stan_2d_main import stan_main
        base_load_data(args)
        stan_main(
            args['trainfeature'],
            args['trainlabel'],
            args['testfeature'],
            args['testlabel'],
            mode='2d',
            epochs=args['epochs'],
            batch_size=args['batch_size'],
            attention_hidden_dim=args['attention_hidden_dim'],
            lr=args['lr'],
            device=args['device']
        )
    elif args['method'] == 'stan':
        from methods.stan.stan_main import stan_main
        base_load_data(args)
        stan_main(
            args['trainfeature'],
            args['trainlabel'],
            args['testfeature'],
            args['testlabel'],
            mode='3d',
            epochs=args['epochs'],
            batch_size=args['batch_size'],
            attention_hidden_dim=args['attention_hidden_dim'],
            lr=args['lr'],
            device=args['device']
        )

    elif args['method'] == 'stagn':
        from methods.stagn.stagn_main import stagn_main, load_stagn_data
        features, labels, g = load_stagn_data(args)
        stagn_main(
            features,
            labels,
            args['test_size'],
            g,
            mode='2d',
            epochs=args['epochs'],
            attention_hidden_dim=args['attention_hidden_dim'],
            lr=args['lr'],
            device=args['device']
        )
    elif args['method'] == 'gtan':
        from methods.gtan.gtan_main import gtan_main, load_gtan_data
        feat_data, labels, train_idx, val_idx, test_idx, g, cat_features, split_meta = load_gtan_data(
            args)
        gtan_main(
            feat_data, g, train_idx, val_idx, test_idx, labels, args, cat_features, split_meta)
    elif args['method'] in {'rgtan', 'rgtan_ca1', 'rgtan_ca1_ca3', 'rgtan_mpfc', 'rgtan_mpfc_vta'}:
        from methods.rgtan.rgtan_main import (
            rgtan_main, rgtan_aml_single_main, rgtan_ca1_main,
            rgtan_ca1_ca3_main, rgtan_mpfc_main, loda_rgtan_data,
        )
        if args.get('dataset') == 'amlsim':
            from datasets.amlsim import load_amlsim_data
            feat_data, labels, train_idx, test_idx, g, cat_features, neigh_features = \
                load_amlsim_data(args)
            nei_key = 'amlsim'
        else:
            feat_data, labels, train_idx, test_idx, g, cat_features, neigh_features = loda_rgtan_data(
                args)
            nei_key = 'aml'
        if args['method'] in ('rgtan_mpfc', 'rgtan_mpfc_vta'):
            rgtan_mpfc_main(feat_data, g, train_idx, args['_val_idx'], test_idx, labels, args,
                            cat_features, neigh_features,
                            nei_att_head=args['nei_att_heads'][nei_key])
        elif args['method'] == 'rgtan_ca1_ca3':
            rgtan_ca1_ca3_main(feat_data, g, train_idx, args['_val_idx'], test_idx, labels, args,
                               cat_features, neigh_features,
                               nei_att_head=args['nei_att_heads'][nei_key])
        elif args['method'] == 'rgtan_ca1':
            rgtan_ca1_main(feat_data, g, train_idx, args['_val_idx'], test_idx, labels, args,
                           cat_features, neigh_features,
                           nei_att_head=args['nei_att_heads'][nei_key])
        elif args['dataset'] == 'aml':
            rgtan_aml_single_main(feat_data, g, train_idx, args['_val_idx'], test_idx, labels, args,
                                  cat_features, neigh_features,
                                  nei_att_head=args['nei_att_heads']['aml'])
        elif args['dataset'] == 'amlsim':
            # AMLSIM baseline — use rgtan_mpfc_main with CA1/CA3/MPFC all disabled in config
            rgtan_mpfc_main(feat_data, g, train_idx, args['_val_idx'], test_idx, labels, args,
                            cat_features, neigh_features,
                            nei_att_head=args['nei_att_heads']['amlsim'])
        else:
            rgtan_main(feat_data, g, train_idx, test_idx, labels, args,
                       cat_features, neigh_features, nei_att_head=args['nei_att_heads'][args['dataset']])
    elif args['method'] == 'hogrl':
        from methods.hogrl.hogrl_main import hogrl_main
        hogrl_main(args)
    else:
        raise NotImplementedError("Unsupported method. ")


if __name__ == "__main__":
    main(parse_args())
