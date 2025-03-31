from structures.instances import Instances
from seq_decoder import SeqDecoder
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from structures.ordered_set import OrderedSet
from collections import deque
from models.feature_embedding import SimpleCNN
from models import GFF
class Pembedding(nn.Module):
    def __init__(self, input_dim=4, output_dim=256):
        super(Pembedding, self).__init__()
        self.embedding = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.embedding(x)

class GIP(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.device = config["DEVICE"]
        self.id_criterion = nn.CrossEntropyLoss()
        self.id_loss_weight = config["ID_LOSS_WEIGHT"]
        self.num_id_vocabulary = config["NUM_ID_VOCABULARY"]
        self.training_num_id = config["NUM_ID_VOCABULARY"] if "TRAINING_NUM_ID" not in config else config[
            "TRAINING_NUM_ID"]
        self.seq_decoder=SeqDecoder(
            detr_hidden_dim=config["DETR_HIDDEN_DIM"],
            hidden_dim=256 if "SEQ_HIDDEN_DIM" not in config else config["SEQ_HIDDEN_DIM"],
            dim_feedforward=512 if "SEQ_DIM_FEEDFORWARD" not in config else config["SEQ_DIM_FEEDFORWARD"],
            num_heads=8 if "SEQ_NUM_HEADS" not in config else config["SEQ_NUM_HEADS"],
            dropout=0.0,
            n_id_decoder_layers=config["ID_DECODER_LAYERS"],
            num_id_vocabulary=self.num_id_vocabulary,
            training_num_id=config["NUM_ID_VOCABULARY"] if "TRAINING_NUM_ID" not in config else config["TRAINING_NUM_ID"],
            device=self.device,
            max_temporal_length=config["MAX_TEMPORAL_LENGTH"],
            multi_times_id_decoder=config["MULTI_TIMES_ID_DECODER"] if "MULTI_TIMES_ID_DECODER" in config else 0,
        )
        if config['DATASETS'] == ['DanceTrack']:
            dataname = 'dance'
        elif config['DATASETS'] == ['SportsMOT']:
            dataname = 'sports'
        elif config['DATASETS'] == ['MOT17'] or ['MOT17' , 'CrowdHuman', 'CrowdHuman']:
            dataname = 'mot'
        else:
            assert False, "Dataset not recognized"
        self.CNN = SimpleCNN(dataname)
        self.p_embedding = Pembedding()
        self.GFF = GFF.GFF().to(self.device)

    def GFFfuse(self,reid1,flow):
      
        return self.GFF(reid1,flow)
        
    def GFFfuse_eval(self,reid1,flow):
        
        self.GFF.eval()
        return self.GFF(reid1,flow)

    
        
        
    
    def add_random_id_words_to_instances(self, instances: list[Instances]):
        # assert len(instances) == 1  # only for bs=1
        ids = torch.cat([instance.ids for instance in instances], dim=0)
        ids_unique = torch.unique(ids)

        if len(ids_unique) > self.training_num_id:
            keep_index = torch.randperm(len(ids_unique))[:self.training_num_id]
            ids_unique = ids_unique[keep_index]
            pass
        id_words_unique = torch.randperm(n=self.num_id_vocabulary)[:len(ids_unique)]
        id_to_word = {
            i.item(): w.item() for i, w in zip(ids_unique, id_words_unique)
        }
        already_id_set = set()
        for t in range(len(instances)):
            id_words, id_labels = [], []
            for _ in range(len(instances[t])):
                i = instances[t].ids[_].item()
                if i in id_to_word:
                    id_words.append(id_to_word[i])
                else:   # handle the case that the number of objects exceeds the length of ID dictionary
                    id_words.append(-1)
                    id_labels.append(-1)
                    continue
                if i in already_id_set:
                    id_labels.append(id_to_word[i])
                else:
                    id_labels.append(self.num_id_vocabulary)
                    already_id_set.add(i)
            instances[t].id_words = torch.tensor(id_words, dtype=torch.long)
            instances[t].id_labels = torch.tensor(id_labels, dtype=torch.long)
            ins_keep_index = instances[t].id_words != -1
            instances[t] = instances[t][ins_keep_index]
        return


    def forward_train(
            self,
            track_history: list[list[Instances]],
            traj_drop_ratio: float,
            traj_switch_ratio: float,
            use_checkpoint: bool = False,
    ):
        assert len(track_history) == 1, f"Only BS=1 is supported."

        pred_id_words, gt_id_words ,id_gts, ap_embed, mask, history= self.seq_decoder(
            track_seqs=track_history,
            traj_drop_ratio=traj_drop_ratio,
            traj_switch_ratio=traj_switch_ratio,
            use_checkpoint=use_checkpoint,
        )

        return pred_id_words, gt_id_words, id_gts, ap_embed, mask, history

    def inference(
            self,
            trajectory_history: deque[Instances],
            num_id_vocabulary: int,
            ids_to_results: dict,
            current_id: int,
            id_deque: OrderedSet,
            id_thresh: float = 0.1,
            newborn_thresh: float = 0.5,
            inference_ensemble: int = 0,
    ):
        """
        :param trajectory_history: Historical trajectories.
        :param num_id_vocabulary: Number of ID vocabulary, K in the paper.
        :param ids_to_results: Mapping from ID word index to ID label in tracker files.
        :param current_id: Current next ID label of tracker files.
        :param id_deque: OrderedSet of ID words, may be recycled.
        :param id_thresh: ID threshold.
        :param newborn_thresh: Newborn threshold,
                               only the conf higher than this threshold will be considered as a newborn target.
        :param inference_ensemble: Ensemble times for inference.
        :return:
        """
        deque_max_length = trajectory_history.maxlen
        trajectory_history_list = list(trajectory_history)
        trajectory = trajectory_history_list[:-1]
        current = trajectory_history_list[-1:]

        # NEED TO KNOW:
        # 1. "ids" is the final ID words for current frame, it is a list.
        #    If a target does not have a corresponding ID word, it will be assigned as -1 in "ids".
        # 2. "new_id" is the ID words that need to be assigned to the new targets, also a list.
        # 3. "current" is the objects in the current frame.

        n_targets_in_frames = [len(_) for _ in trajectory_history_list]
        num_history_tokens, num_current_tokens = sum(n_targets_in_frames[:-1]), sum(n_targets_in_frames[-1:])
        if num_history_tokens == 0:     # no history tokens
            ids = [-1] * num_current_tokens
        elif num_current_tokens == 0:   # no current tokens
            ids = []
            return ids, trajectory_history, ids_to_results, current_id, id_deque, None  # directly return
        else:  # normal process:
            trajectory_id_set = set(torch.cat([_.ids for _ in trajectory_history_list[:-1]], dim=0).cpu().tolist())
            # Seq Decoding:
            pred_id_words, _,_,_,_,_ = self.seq_decoder(
                track_seqs=[trajectory_history_list],
                inference_ensemble=inference_ensemble,
            )
            if isinstance(pred_id_words, torch.Tensor):
                id_confs = torch.softmax(pred_id_words, dim=-1)  # [1, N, K + 1]
                id_confs = id_confs[0]  # [N, K + 1]
            else:
                assert isinstance(pred_id_words, list)
                # id_confs = [torch.softmax(_, dim=1) for _ in pred_id_words]
                id_confs = [_ for _ in pred_id_words]
                id_confs = [_[0] for _ in id_confs]
                _ensemble_n = len(id_confs)
                id_confs = torch.stack(id_confs, dim=0)  # [T, N, K + 1]
                id_confs = torch.sum(id_confs, dim=0)  # [N, K + 1]
                id_confs = id_confs / _ensemble_n
                id_confs = torch.softmax(id_confs, dim=-1)  # [N, K + 1]
                pass

            ids = list()
            newborn_repeat = id_confs[:, -1:].repeat(1, len(id_confs) - 1)
            extended_id_confs = torch.cat((id_confs, newborn_repeat), dim=-1)
            match_rows, match_cols = linear_sum_assignment(1 - extended_id_confs.cpu())
            for _ in range(len(match_rows)):
                _id = match_cols[_]
                if _id not in trajectory_id_set:
                    ids.append(-1)
                elif _id >= num_id_vocabulary:
                    ids.append(-1)
                elif id_confs[match_rows[_], _id].item() < id_thresh:
                    ids.append(-1)
                else:
                    ids.append(_id)


        # Update the ID deque:
        for _id in ids:
            if _id != -1:
                id_deque.add(_id)

        # Filter the newborn targets, True means marked as newborn but not reach the newborn threshold:
        newborn_neg_filter = ((torch.tensor(ids).to(current[0].confs.device) == -1)
                              & (current[0].confs <= newborn_thresh).reshape(-1, ))

        if torch.sum(~newborn_neg_filter) > num_id_vocabulary:
            # The legal objects are too many, we need to filter out some of them.
            # Warning: This should not happen in normal cases.
            #          If it happens, you may increase the ID vocabulary size.
            print(f"[Warning!] There are too many objects, N={torch.sum(~newborn_neg_filter)}. ")
            already_ids_num = torch.sum(torch.tensor(ids) != -1)
            newborn_index = torch.tensor(ids).to(current[0].confs.device) == -1
            confs = current[0].confs.reshape(-1, ) * newborn_index.to(float)
            newborn_num_in_legal = num_id_vocabulary - already_ids_num
            index = torch.topk(confs, k=newborn_num_in_legal, dim=0).indices
            newborn_neg_filter_from_topk = torch.tensor(ids).to(current[0].confs.device) == -1
            newborn_neg_filter_from_topk[index] = False
            legal_newborn_neg_filter = newborn_neg_filter | newborn_neg_filter_from_topk
            newborn_neg_filter = legal_newborn_neg_filter
            print(f"[Warning!] Because the newborn objects are too many, "
                  f"we only keep {newborn_num_in_legal} newborn objects with highest confs. "
                  f"Already assigned {already_ids_num} IDs. "
                  f"Now we have {torch.sum(~newborn_neg_filter)} IDs.")

        # Just a check!
        assert torch.sum(
            ~newborn_neg_filter) <= num_id_vocabulary, f"Too many IDs: {torch.sum(~newborn_neg_filter)}."

        # Remove the illegal newborn targets (conf < newborn_thresh):
        ids = torch.tensor(ids)[~newborn_neg_filter.cpu()].tolist()
        current[0] = current[0][~newborn_neg_filter]

        num_new_id = ids.count(-1)  # how many new ID words need to be assigned

        if num_new_id > 0:  # assign new ID words
            id_deque_list = list(id_deque)
            if len(id_deque_list) + num_new_id <= num_id_vocabulary:
                # The ID dictionary is not fully used, we can directly assign new ID words.
                new_ids = [len(id_deque_list) + _ for _ in range(num_new_id)]  # ID dictionary index (ID words)
            else:
                # The ID dictionary is fully used, we need to recycle some ID words.
                if len(id_deque_list) < num_id_vocabulary:
                    # There are still some empty slots in the ID dictionary,
                    # we can directly assign these clear_id_num_new_id new ID words.
                    clear_num_new_id = num_id_vocabulary - len(id_deque_list)
                    conflict_num_new_id = num_new_id - clear_num_new_id
                    new_ids = [len(id_deque_list) + _ for _ in range(clear_num_new_id)]
                else:
                    # There are no empty slots in the ID dictionary,
                    # we need to recycle conflict_num_new_id ID words.
                    conflict_num_new_id = num_new_id
                    new_ids = []
                # Recycled ID words:
                conflict_new_id = id_deque_list[:conflict_num_new_id]
                # As we need to recycle some ID words in conflict_new_id,
                # we need to remove the corresponding tracklets in the trajectory.
                for _ in range(len(trajectory)):
                    conflict_index = torch.zeros([len(trajectory[_]), ], dtype=torch.bool,
                                                 device=trajectory[_].ids.device)  # init
                    for _id in conflict_new_id:
                        conflict_index = conflict_index | (trajectory[_].ids == _id)
                    trajectory[_] = trajectory[_][~conflict_index]
                new_ids = new_ids + conflict_new_id  # assign the recycled ID words to "new_ids"

            # Update the corresponding mapping from ID words to ID labels (in tracker outputs):
            for _id in new_ids:
                ids_to_results[_id] = current_id
                current_id += 1
                id_deque.add(_id)

            # Insert the new_ids into the ids list:
            new_id_idx = 0
            ori_ids = ids
            ids = []
            for _ in ori_ids:
                if _ == -1:  # new id need to add:
                    ids.append(new_ids[new_id_idx])
                    new_id_idx += 1
                else:
                    ids.append(_)

        current[0].ids = torch.tensor(ids, dtype=torch.long, device=current[0].ids.device)
        trajectory_history_list = trajectory + current
        trajectory_history = deque(trajectory_history_list, maxlen=deque_max_length)
        assert len(ids) == len(set(ids)), f"ids is not unique: ids={ids}."
        return ids, trajectory_history, ids_to_results, current_id, id_deque, ~newborn_neg_filter
        # We will remove some illegal newborn targets in the outer function,
        # based on the "newborn_neg_filter" flags.

def build(config: dict):
    return GIP(config=config)

if __name__ == "__main__":
    config={}
    model = GIP(config)
    for name, param in model.named_parameters():
        print(f"Parameter Name: {name}, Parameter Shape: {param.shape}")