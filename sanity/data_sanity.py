from consts import DEF_DATA_PATH
from data.dataset_utils import GWilliams, random_split, Armeni
from train.parser_utils import get_base_parser


def test_no_time_overlap_sess_exclusion(data_path):
    parser = get_base_parser()
    args = parser.parse_args()

    args.data_path = data_path

    args.exclude_subj_sess_tasks = [('001', '002', 'compr')]
    dataset = Armeni(subjects=['001'], sessions=['001', '002'], tasks=['compr'], args=args)
    train_set, val_set, test_set = random_split(dataset, args, *[0.3, 0.3, 0.4])

    # ensure each time appears only once
    def collect_times(data_subset):
        return [time for i in range(len(data_subset)) for time in data_subset[i]['t']]

    train_times, val_times, test_times = map(collect_times, [train_set, val_set, test_set])

    assert (len(train_times) == len(set(train_times)) and
            len(val_times) == len(set(val_times)) and
            len(test_times) == len(set(test_times))), "Multiple instances of the same time in the same split"

    train_times, val_times, test_times = map(set, [train_times, val_times, test_times])

    assert (train_times.isdisjoint(val_times) and
            train_times.isdisjoint(test_times) and
            val_times.isdisjoint(test_times)), "Datasets overlap in time"


def assert_no_time_overlap(args):
    dataset = GWilliams(subjects=['01'], sessions=['0'], tasks=['0'], args=args)
    train_set, val_set, test_set = random_split(dataset, args, *[0.3, 0.3, 0.4])

    def collect_times(data_subset):
        return {time for i in range(len(data_subset)) for time in data_subset[i]['t']}

    train_times, val_times, test_times = map(collect_times, [train_set, val_set, test_set])

    assert (train_times.isdisjoint(val_times) and
            train_times.isdisjoint(test_times) and
            val_times.isdisjoint(test_times)), "Datasets overlap in time"


def test_dataset_split_no_time_overlap_no_context(data_path):
    parser = get_base_parser()
    args = parser.parse_args()

    args.data_path = data_path

    args.right_context = 0
    args.left_context = 0

    assert_no_time_overlap(args)


def test_dataset_split_no_time_overlap_with_context(data_path):
    parser = get_base_parser()
    args = parser.parse_args()

    args.data_path = data_path

    args.right_context = 2
    args.left_context = 2

    assert_no_time_overlap(args)


if __name__ == '__main__':
    data_path = DEF_DATA_PATH

    test_no_time_overlap_sess_exclusion(data_path)
    test_dataset_split_no_time_overlap_no_context(data_path)
    test_dataset_split_no_time_overlap_with_context(data_path)
