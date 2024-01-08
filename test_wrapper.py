from tqdm.auto import tqdm


def wrapper(gen, key_list, len_db):
    for idx, item in enumerate(tqdm(key_list, desc=f'Reading steps', total=len_db)):
        try:
            yield gen[item]
        except StopIteration:
            break
        except Exception as e:
            print(f'Unable to load key {idx}: {e.__class__}: {str(e)}')
            # continue
            pass


test_dict = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7, 'i': 8, 'j': 9, 'k': 10, 'l': 11, 'm': 12,
             'n': 13, 'o': 14, 'p': 15, 'q': 16, 'r': 17, 's': 18, 't': 19, 'u': 20, 'v': 21, 'w': 22, 'x': 23, 'y': 24,
             'z': 25}
test_list = ['a', 'b', 'c', 'd', 'e', 'f', 'ae', 'xy', 'ou', 'i', 'j', 'k', 'oe', 'l', 'm', 'n', 'o', 'ue', 'p', 'q', 'r',
             's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
n_keys = len(test_list)

keys = wrapper(test_dict, test_list, n_keys)
for key in keys:
    print(key)

