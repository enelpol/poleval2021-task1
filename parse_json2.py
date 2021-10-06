import json
import re
import sys




def load_json(path):
    data = json.load(open(path))

    text = ''
    text_in=''
    text_exp=''
    for token in data['words']:
        text += token['word'] + token['punctuation']
        # text_exp+=token['word'] + token['punctuation']
        if re.match('^[^\w"%]+$',token['word']):
            pass    
            
        else:
            text_in += token['word']

            # print(token)
            if token['punctuation']=='-' and token['space_after']==False:
                text_exp += token['word']
            else:
                text_exp += token['word'] + token['punctuation']
                # print(token['punctuation'])
        
        # if token['space_after']:
        #     text += ' '        
        if token['space_after'] or token['punctuation']!='':
            text_in += ' '

        if token['space_after'] or token['punctuation']!='':

            text_exp+=' '
        text += ' '
        
    
    text_in = text_in.lower()
    text_in = re.sub('"',' " ',text_in)
    text_in = re.sub('\+',' + ',text_in)

    text_in=re.sub('[,!?.:;-]',' ',text_in)

    text_in = re.sub(' +', ' ', text_in)

    text = text.lower()
    # text=text.replace('-',' ')
    text=re.sub('(\? )+','? ',text)
    text=re.sub('(! )+','! ',text)
    text=re.sub('"-','"',text)
    text=re.sub(' +',' ',text)

    text_exp = text_exp.lower()
    text_exp = re.sub('\.\.\.', '…', text_exp)
    text_exp = re.sub('"', ' " ', text_exp)
    text_exp = re.sub('&quot;', ' " ', text_exp)
    text_exp = re.sub('&apos;', '\'', text_exp)
    text_exp = re.sub('&#93;', '', text_exp) #]
    text_exp = re.sub('&#91;', '', text_exp) #[
    text_exp = re.sub('\+', ' + ', text_exp)
    text_exp = re.sub(r' ([…,!?.:;-])', '\\1', text_exp)
    #TODO zamienić powyższe znaki na spacje jeśli po nich nie ma spacji?
    text_exp = re.sub(r'[…,!?.:;-]([^ ])', ' \\1', text_exp) #usuwa trzy kropki - może zamienić na elipsis
    #pojedyncze znaki
    text_exp = re.sub(r' […,!?.:;-] ', ' ', text_exp)
    
    text_exp = re.sub(' +', ' ', text_exp)
    # print(text_exp)
    
    text_in=text_exp
    text_in = re.sub('([^ ])[…,!?.:;-]( |$)', '\\1 ', text_in)
    text_in = re.sub(' +', ' ', text_in)

    text_exp = re.sub('…', '...', text_exp)
    
    try:
        assert len(text_in.strip().split(' '))==len(text_exp.strip().split(' '))
    except AssertionError:
        print(len(text_in.strip().split(' ')),len(text_exp.strip().split(' ')), file=sys.stderr)

        print(text_exp.strip().split(' '), file=sys.stderr)
        print(text_in.strip().split(' '), file=sys.stderr)
        for a,b in zip(text_exp.strip().split(' '), text_in.strip().split(' ')):
            print(a,b, file=sys.stderr)
        print()
    
    return text_in.strip(), text_exp.strip()

if __name__ == "__main__":
    train_path = '../2021-punctuation-restoration/train/'
    rest_in=open('rest_in.tsv','w')
    rest_expected=open('rest_expected.tsv','w')
    for in_line, expected_line in zip(open(train_path+'in.tsv'), open(train_path+'expected.tsv')):
        if in_line[-1]=='\n': in_line=in_line[:-1]
        if expected_line[-1]=='\n': expected=expected_line[:-1]
        name, text = in_line.split('\t')
        if name.startswith('wikinews'):
            json_in, json_expected=load_json(f'../rest_wikinews_all/json/{name}.json')
        elif name.startswith('wikitalk'):
            json_in, json_expected=load_json(f'../rest_wikitalk_all/json/{name}.json')
        
        rest_in.write(f'{name}\t{json_in}\n')
        rest_expected.write(f'{json_expected}\n')
    