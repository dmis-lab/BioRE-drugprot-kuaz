import json
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--output_path', type=str,  help='')
parser.add_argument('--bc7format_out_path', type=str,  help='')
parser.add_argument('--mapping_path', type=str,  help='')
parser.add_argument('--label_path', type=str,  help='Path to json dict. {"label name":int type order}')
parser.add_argument('--task', type=str,  default="bc7dp", help='default:bc7dp, possible other options:{"bc7dp", "bc7dp-None"}. if "bc7dp-None", calcuated score includes None labels as positive class.')
args = parser.parse_args()


bc7outdf = pd.read_csv(args.mapping_path, sep="\t", index_col=0)
preddf = pd.read_csv(args.output_path, sep="\t", index_col=0)

label_dict = json.load(open(args.label_path, 'r'))
rev_label_dict = {value:key for key, value in label_dict.items()}

if args.task.lower() in ["bc7dp","bc7dp-none"]:
    pred = [preddf.iloc[i].tolist() for i in preddf.index]

    labels = [val[1] for val in sorted(rev_label_dict.items())]
        
    pred_class = preddf['prediction'].to_list()
    
    if args.task == "bc7dp":
        pos_labels = [lab for lab in labels if lab!="NONE"]
        #count_P_NONE = sum([1 for lab in pred_class if lab == "NONE"])
        #print(f"The number of negative samples (NONE) in prediction = {count_P_NONE}")
    elif args.task.lower() == "bc7dp-none":
        pos_labels = labels

    assert len(bc7outdf) == len(pred_class)
    bc7outdf['relation'] = pred_class
    bc7outdf.index = [ele.split("_")[0] for ele in bc7outdf.index]
    bc7outdf.astype(str).to_csv(args.bc7format_out_path, sep='\t', header=False, index=True)

else:
    raise NotImplemented
