import numpy as np
import torch


def get_distances_hypro(pred_dt, pred_type_result, gt_dt, gt_type_result, num_classes, filter, time_range,
                        distance_del_cost, trans_cost):
    '''

    :param pred_dt:
    :param pred_type_result:
    :param gt_dt:
    :param gt_type_result:
    :param num_classes:
    :param filter:
    :param time_range:
    :param distance_del_cost:
    :param trans_cost:
    :return:
    '''
    distances = []

    for pred_time, pred_type, gt_time, gt_type in zip(pred_dt, pred_type_result, gt_dt, gt_type_result):

        ref_seq = [pred_time.cpu(), pred_type.cpu()]
        decode_seq = [gt_time.cpu(), gt_type.cpu()]

        if filter:
            decode_seq, ref_seq = filter_points(decode_seq, ref_seq, time_range)

        distance = distance_between_event_seq(ref_seq, decode_seq,
                                              distance_del_cost, trans_cost, num_classes)[0]
        distances.append(distance)
    return distances


def get_distances_diffusion(pred_dt, pred_type_result, gt_dt, gt_type_result, num_classes, filter, time_range,
                            distance_del_cost, trans_cost):
    '''
    Diffusion version
    This is for diffusion, a little bit different from HYPRO version, since HYPRO does not provide dt seqs it only
    provides time stamps seqs. Therefore, it does not need these two lines for HYPRO but needed for diffusion version
    'pred_time = torch.cumsum(pred_time, dim=-1)'
    'gt_time = torch.cumsum(gt_time, dim=-1)'
    For param, see type_rmse_diffusion
    :param pred_dt:
    :param pred_type_result:
    :param gt_dt:
    :param gt_type_result:
    :param num_classes:
    :param filter:
    :param time_range:
    :param distance_del_cost:
    :param trans_cost:
    :return:
    '''

    distances = []

    for pred_time, pred_type, gt_time, gt_type in zip(pred_dt, pred_type_result, gt_dt, gt_type_result):

        pred_time = torch.cumsum(pred_time, dim=-1)
        gt_time = torch.cumsum(gt_time, dim=-1)

        ref_seq = [pred_time.cpu(), pred_type.cpu()]
        decode_seq = [gt_time.cpu(), gt_type.cpu()]

        if filter:
            decode_seq, ref_seq = filter_points(decode_seq, ref_seq, time_range)

        distance = distance_between_event_seq(ref_seq, decode_seq,
                                              distance_del_cost, trans_cost, num_classes)[0]
        distances.append(distance)

    return distances


def type_rmse_hypro(pred_dt, pred_type_result, gt_dt, gt_type_result, num_classes, filter, time_range, **kwargs):
    '''
    Type RMSE see document
    HYPRO version
    This is for hypro, a little bit different from diffusion version, since HYPRO does not provide dt seqs it only
    provides time stamps seqs. Therefore, it does not need these two lines for HYPRO
    'pred_time = torch.cumsum(pred_time, dim=-1)'
    'gt_time = torch.cumsum(gt_time, dim=-1)'
    :param pred_dt: B x Seq_Len
    :param pred_type_result: B x Seq_Len
    :param gt_dt: B x Seq_Len
    :param gt_type_result: B x Seq_Len
    :param num_classes:
    :param filter: filter the seq to meet the time range or not
    :param time_range: time range for filtering
    :param kwargs:
    :return:
    '''

    gt_type_count = torch.zeros(num_classes)
    pred_type_count = torch.zeros(num_classes)

    rmse_types = []
    for pred_time, pred_type, gt_time, gt_type in zip(pred_dt, pred_type_result, gt_dt, gt_type_result):

        # pred_time = torch.cumsum(pred_time, dim=-1)
        # gt_time = torch.cumsum(gt_time, dim=-1)

        ref_seq = [pred_time.cpu(), pred_type.cpu()]
        decode_seq = [gt_time.cpu(), gt_type.cpu()]
        if filter:
            decode_seq, ref_seq = filter_points(decode_seq, ref_seq, time_range)

        gt_type = torch.tensor(decode_seq[1])
        pred_type = torch.tensor(ref_seq[1])
        for i in range(num_classes):
            gt_type_count[i] = gt_type[gt_type == i].size(0)
            pred_type_count[i] = pred_type[pred_type == i].size(0)
        rmse_types.append(torch.sqrt(((pred_type_count - gt_type_count) * (pred_type_count - gt_type_count)).mean()))
    return rmse_types


def type_rmse_diffusion(pred_dt, pred_type_result, gt_dt, gt_type_result, num_classes, filter, time_range, **kwargs):
    '''
    Type RMSE see document
    Diffusion version
    This is for diffusion, a little bit different from HYPRO version, since HYPRO does not provide dt seqs it only
    provides time stamps seqs. Therefore, it does not need these two lines for HYPRO but needed for diffusion version
    'pred_time = torch.cumsum(pred_time, dim=-1)'
    'gt_time = torch.cumsum(gt_time, dim=-1)'
    :param pred_dt: B x Seq_Len
    :param pred_type_result: B x Seq_Len
    :param gt_dt: B x Seq_Len
    :param gt_type_result: B x Seq_Len
    :param num_classes:
    :param filter: filter the seq to meet the time range or not
    :param time_range: time range for filtering
    :param kwargs:
    :return:
    '''

    gt_type_count = torch.zeros(num_classes)
    pred_type_count = torch.zeros(num_classes)

    rmse_types = []
    for pred_time, pred_type, gt_time, gt_type in zip(pred_dt, pred_type_result, gt_dt, gt_type_result):

        pred_time = torch.cumsum(pred_time, dim=-1)
        gt_time = torch.cumsum(gt_time, dim=-1)

        ref_seq = [pred_time.cpu(), pred_type.cpu()]
        decode_seq = [gt_time.cpu(), gt_type.cpu()]
        if filter:
            decode_seq, ref_seq = filter_points(decode_seq, ref_seq, time_range)

        gt_type = decode_seq[1].detach().clone()  #gt_type = torch.tensor(decode_seq[1])
        pred_type = ref_seq[1].detach().clone()  #pred_type = torch.tensor(ref_seq[1])
        for i in range(num_classes):
            gt_type_count[i] = gt_type[gt_type == i].size(0)
            pred_type_count[i] = pred_type[pred_type == i].size(0)
        rmse_types.append(torch.sqrt(((pred_type_count - gt_type_count) * (pred_type_count - gt_type_count)).mean()))
    return rmse_types



# ref: https://github.com/hongyuanmei/neural-hawkes-particle-smoothing/blob/8f33c75038e739a2a0b61db854dd97d918ce2d19/nhps/distance/utils/edit_distance.py
def find_alignment_mc(seq1, seq2, del_cost, trans_cost):
    """
    We use dynamic programming to find the best alignments between two seqs.
    ``nc'' means that this functions support a series of del_cost values.
    Note: Not support multiple types.
    :param np.ndarray seq1: Time stamps of seq #1.
    :param np.ndarray seq2: Time stamps of seq #2.
    :param np.ndarray del_cost: A series of delete cost.
    :param float trans_cost: Transportation cost per unit length.
    :return: Alignment list and minimum distances for all the del_cost values.
    """
    n_cost = len(del_cost)
    n1 = len(seq1)
    n2 = len(seq2)
    # shape=[n2, n1]
    trans_mask = np.abs(seq2.repeat(n1).reshape(n2, n1) - seq1) * trans_cost
    # shape=[n1+1, n1+1]
    del_mask = np.arange(n1 + 2, dtype=np.float32) \
                   .repeat(n1 + 1).reshape(n1 + 2, n1 + 1) \
                   .T.reshape(-1)[:(n1 + 1) ** 2].reshape(n1 + 1, n1 + 1) - 1
    del_mask[np.tril_indices(n1 + 1, -1)] = float('inf')
    # shape=[n1+1, n1+1, n_cost]
    del_mask = del_mask.repeat(n_cost).reshape(n1 + 1, n1 + 1, n_cost) * del_cost
    # shape=[n1+1, n1+1, n_cost]
    del_mask = del_mask.transpose([1, 0, 2]).copy()
    # shape=[n1+1, n_cost]
    overhead = np.empty(shape=[n1 + 1, n_cost], dtype=np.float32)
    overhead.fill(float('inf'))
    overhead[0, :] = 0.0
    # shape=[n2, n1+1, n_cost]
    back_pointers = np.empty(shape=[n2, n1 + 1, n_cost], dtype=np.int32)
    for n2_idx in range(n2):
        # shape=[n1+1, n1+1, n_cost]
        add_mask = del_mask.copy()
        add_mask[1:, :, :] += np.outer(trans_mask[n2_idx],
                                       np.ones(shape=[(n1 + 1) * n_cost],
                                               dtype=np.float32)).reshape(n1, n1 + 1, n_cost)
        add_mask[np.arange(n1 + 1), np.arange(n1 + 1), :] = del_cost
        # shape=[n1+1, n1+1, n_cost]
        cost_mat = overhead + add_mask
        # shape=[n1+1, n_cost]
        choices = np.argmin(cost_mat, axis=1)
        back_pointers[n2_idx] = choices
        overhead = cost_mat.min(axis=1)
    overhead += np.outer(np.arange(n1, -1, -1, dtype=np.float32), np.ones(shape=[n_cost])) * del_cost
    # shape=[n_cost]
    curr_choice = np.argmin(overhead, axis=0)
    # shape=[n_cost]
    min_distance = overhead.min(axis=0)
    best_route = [curr_choice]
    # shape=[n1+1, n_cost]
    for choice_list in back_pointers[::-1]:
        # shape=[n_cost]
        curr_choice = choice_list[curr_choice, np.arange(n_cost)]
        best_route.append(curr_choice)
    # shape=[n2, n_cost]
    best_route = np.array(best_route)

    align_pairs = list()
    for cost_idx in range(n_cost):
        best_route_ = best_route[:, cost_idx]
        pairs = list()
        memo = -1
        for n2_idx_plus_1, choice_made in enumerate(best_route_[::-1]):
            if choice_made != memo:
                pairs.append([choice_made - 1, n2_idx_plus_1 - 1])
            memo = choice_made
        align_pairs.append(pairs[1:])

    return [align_pairs,  # len=n_cost
            min_distance  # shape=[n_cost]
            ]


def find_alignment(seq1, seq2, del_cost, trans_factor):
    """
    Similar functionality with find_alignment_nc, but for single del_cost cost.
    :param np.ndarray seq1:
    :param np.ndarray seq2:
    :param float del_cost:
    :param float trans_factor:
    :return:
    """
    align_pairs, min_distance = \
        find_alignment_mc(seq1, seq2, np.array([del_cost]), trans_factor)
    return align_pairs[0], float(min_distance[0])



def filter_points(ground_truth_tuple, sample_tuple, time_range):
    # filter_max_time = ground_truth_tuple[0][0] + time_range
    filter_max_time = time_range
    horizon = len(ground_truth_tuple[0])

    def _truncate_tuple(one_tuple):
        end_i = horizon
        for i in range(horizon):
            if one_tuple[0][i] > filter_max_time:
                end_i = i
                break

        return one_tuple[0][:end_i], one_tuple[1][:end_i]

    # filter ground truth
    ground_truth_tuple = _truncate_tuple(ground_truth_tuple)
    sample_tuple = _truncate_tuple(sample_tuple)

    return ground_truth_tuple, sample_tuple


def distance_between_event_seq(ref_seq, decode_seq, del_cost, trans_cost, num_types):
    """
    Args:
        ref_seq: [time_seqs, event_seqs]
        decode_seq: [time_seqs, event_seqs]
        del_cost:
        trans_cost:
        num_types:

    Returns:

    """
    num_cost = len(del_cost)
    distances = np.zeros(shape=[num_cost], dtype=np.float32)
    total_trans_cost = np.zeros(shape=[num_cost], dtype=np.float32)
    num_true = np.zeros(shape=[num_cost], dtype=np.int32)
    num_del = np.zeros(shape=[num_cost], dtype=np.int32)
    num_ins = np.zeros(shape=[num_cost], dtype=np.int32)
    num_align = np.zeros(shape=[num_cost], dtype=np.int32)

    seq_per_types = [[list(), list()] for _ in range(num_types)]
    for seq_idx, seq in enumerate([ref_seq, decode_seq]):
        for event_time, event_type in zip(*seq):
            if event_type >= num_types:
                continue
            seq_per_types[event_type][seq_idx].append(event_time)

    for type_idx in range(num_types):
        ref_time = np.array(seq_per_types[type_idx][0])
        decoded_time = np.array(seq_per_types[type_idx][1])
        align_pairs, min_distance = find_alignment_mc(
            ref_time, decoded_time, del_cost, trans_cost)
        for cost_idx in range(num_cost):
            align_pairs_per_cost = align_pairs[cost_idx]
            min_distance_per_cost = min_distance[cost_idx]
            num_align[cost_idx] += len(align_pairs_per_cost)
            num_true[cost_idx] += len(ref_time)
            n_ins_per_cost = len(decoded_time) - len(align_pairs_per_cost)
            n_del_per_cost = len(ref_time) - len(align_pairs_per_cost)
            num_ins[cost_idx] += n_ins_per_cost
            num_del[cost_idx] += n_del_per_cost
            distances[cost_idx] += min_distance_per_cost
            total_trans_cost[cost_idx] += min_distance_per_cost \
                                          - del_cost[cost_idx] * (n_ins_per_cost + n_del_per_cost)
    return distances, total_trans_cost, num_true, num_del, num_ins, num_align


def time_rmse_tensor(preds, labels, **kwargs):
    dt_pred = preds
    dt_label = labels
    rmse = torch.sqrt(torch.mean((dt_pred - dt_label) ** 2, dim=-1))
    rmse_mean = torch.mean(rmse)
    rmse_std = torch.std(rmse)
    return rmse_mean, rmse_std

def sMape_tensor(preds, labels, **kwargs):
    dt_pred = preds
    dt_label = labels + 1e-9
    mae = torch.mean((torch.abs(dt_pred - dt_label) / (torch.abs(dt_label) + torch.abs(dt_pred))), dim=-1) * 200
    smape_mean = torch.mean(mae)
    smape_std = torch.std(mae)
    return smape_mean, smape_std