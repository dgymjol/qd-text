import torch
import numpy as np
from os.path import join
from run_on_video.data_utils import ClipFeatureExtractor
import json
from sys import argv

def main(query_json_file, clip_model_name_or_path='ViT-B/32', clip_len=2):

    print("Loading feature extractors {} ...".format(clip_model_name_or_path))

    feature_extractor = ClipFeatureExtractor(
        framerate=1/clip_len, size=224, centercrop=True,
        model_name_or_path=clip_model_name_or_path, device='cuda'
    )


    for split in ['train', 'val', 'test']:
        query_json_file
        with open(join(query_json_file, f'{split}_output.json'), 'r') as f: 
            query_json = json.load(f)

        with torch.no_grad():
            for qid in query_json.keys():
                query_list = [query_json[qid]]
                q_feat = feature_extractor.encode_text(query_list)[0].detach().cpu().numpy()  # #text * (L, d)

                np.savez(join(query_json_file,"features/qid{}.npz".format(qid)), last_hidden_state=q_feat)
                print("save query features to {}".format(join(query_json_file,"features/qid{}.npz".format(qid))))
            

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')

    query_json_file = argv[1]
    print("==========================================")
    print("query_json_file: ", query_json_file)
    print("==========================================")

    main(query_json_file)