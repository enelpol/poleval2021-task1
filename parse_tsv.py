import sys
from argparse import ArgumentParser

def read_times_data(path):
    data = []
    for line in open(path):
        if line[-1] == '\n': line = line[:-1]
        if line == '</s>': continue
        # try:
        times, text = line.split(' ')
        # except:
        #     print(line)
        start, end = times[1:-1].split(',')
        start = int(start)
        end = int(end)

        data.append((start, end, text))
    return data

def times_after_tokens(path):
    data = read_times_data(path)
    result=[]
    for i in range(len(data)):
        start, end, text = data[i]
        label = text[-1] if text[-1] in ',!?.:;-' else 'B'

        try:
            przerwa = data[i + 1][0] - end
        except IndexError:  # ostatni znak
            przerwa = 0
        # print(label, przerwa)
        result.append((text, przerwa))
    return result

def match_times(times, expected):
    # for token, time in zip(expected, times):
    #     print(token, time)
    # print()
    
    # print(''.join(expected).lower())
    # print(''.join([x[0] for x in times]).lower())
    # assert ''.join(expected).lower()==''.join([x[0] for x in times].lower())
    matched=[]
    
    
    times_text=''
    times_indexes={}
    for token, time in times:
        times_indexes[len(times_text)]=time
        times_text+=token.lower()

    # print(times_text)
    # print(times_indexes)

    index = 0
    for token in expected:
        # print(token)
        found_index=times_text.find(token.lower(), index)
        if found_index>=0:
            if found_index in times_indexes:
                matched.append(times_indexes[found_index])
                index=found_index
            else:
                # print('E1', token)
                matched.append(-1)
        else:
            # print('E2', token)
            matched.append(-1)
    return matched

if __name__ == "__main__":
    parser = ArgumentParser(description='convert text in tsv to CONLL')
    # parser.add_argument('path', help='path to directory with in.tsv and expected.tsv')
    parser.add_argument('in_path', help='path to in.tsv')
    parser.add_argument('expected_path', help='path to expected.tsv')
    parser.add_argument('--times', default=None, help='path to dir with *.clntmstmp')

    parser.add_argument('save_path', help='path to input')
    args = parser.parse_args()

    out = open(args.save_path, 'w')

    for in_line, expected in zip(open(args.in_path), open(args.expected_path)):
        if in_line[-1]=='\n': in_line = in_line[:-1]
        name, text = in_line.split('\t')

        if expected[-1]=='\n': expected = expected[:-1]

        assert len(text.split(' ')) == len(expected.split(' '))

        if args.times:
            times = times_after_tokens(args.times+f'/{name}.clntmstmp')
            # print(len(times), len(text.split(' ')))
            matched=match_times(times, expected.split(' '))
            # for in_token, time in zip(text.split(' '), times):
            #     print(in_token, time)
            # print()
            
        
        for i, (in_token, expected_token) in enumerate(zip(text.split(' '), expected.split(' '))):
            # print(in_token, expected_token)
            expected_token = expected_token.lower()
            # print(in_token, expected_token)
            if in_token == expected_token:
                label = 'B'
            else:
                if in_token == expected_token[:-1]:
                    label = expected_token[-1]
                elif in_token == expected_token[:-3] and expected_token[-3:] == '...':
                    label = expected_token[-3:]
                    
                else:
                    print('ERROR', in_token, expected_token, file=sys.stderr)

            if args.times:
                out.write(f'{in_token}\t{label}\t{matched[i]}\n')
            else:
                out.write(f'{in_token}\t{label}\n')
        out.write('\n')