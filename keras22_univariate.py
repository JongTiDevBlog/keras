from numpy import array

def split_sequence(sequence, n_steps):
    x, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence)-1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        x.append(seq_x)
        y.append(seq_y)
    return array(x), array(y)

dataset = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
n_steps = 3
x, y = split_sequence(dataset, n_steps)

print(x)
print(y)
'''
sequence = 10, n_steps = 4 라면
0,1,2,3/4
1,2,3,4/5

5,6,7,8/9
'''