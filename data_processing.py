

from datasets import load_dataset  # this is from huggingface
import os

def filter_complex_lines(dataset, query_length_limit=200, question_length_limit=120, filtering_keywords=None):
    '''
    Filter out complex entries from the dataset, including the following
    1. querys which contains more than {query_length_limit} chars
    2. questions longer than {question_length_limit} 
    3. queries containing: {filtering_keywords}
    4. queries with subqueries (multiple nested SELECTs), because they are too difficult for LLMs to learn well

    return:
        dict: dataset after the filteration process.
    '''

    # filtering out complex queries defined as follows, only the most basic queries are left as training materials.
    if filtering_keywords is None:
        filtering_keywords = ['join', 'group by', 'order by', 'intersect', 'except', 'union', 'having']

    print('--------------------Starting filtering process --------------')

    for line in dataset:

        if len(line)<=0: # incase empty lines
            continue
        
        query = line['answer'].lower()
        question = line['question'].lower()

        if (len(query) > query_length_limit) or (len(question) > question_length_limit):
            continue  # filter long query and long question

        filter_keyword_flag = False  # filter keyword
        for keyword in filtering_keywords:
            if keyword in query:
                filter_keyword_flag = True
                break
        if filter_keyword_flag:
            continue

        if query.count('select') > 1: # avoid multiple select
            continue
        
        yield line

def format_n_save(dataset, file_path):  # save as file with certain format (context+question+sql)
    cnt = 0

    with open(file_path, 'w', encoding='utf-8') as f:

        for line in dataset:
            context = line['context'].strip()
            question = line['question'].strip()
            query = line['answer'].strip()
            formatted_string = "[context] " + str(context) + " [question] " + str(question) + " [SQL] " + str(query)
            f.write(formatted_string + '\n')
            cnt += 1

            # if cnt>500:
            #     break

    # print(f'After filteration there are {cnt} lines remaining, wrote to path {file_path}')


def process_and_save_sql_data(fp_in=None, test_size=0.1, fp_out=None):
    '''
    data preprocessing pipleline: read-filter-format-save2file
    test_size (float): proportion of test split.
    '''

    if fp_in is None:
        fp_in = 'b-mc2/sql-create-context'
        if not os.path.exists(fp_in):
            raise ValueError('fp does not exist')

    try:
        dataset = load_dataset(fp_in, split='train')
    except Exception as e:
        print(f'Failed to load the dataset. Error: {e}')
        return

    filtered_list = list(filter_complex_lines(dataset))
    # print(len(filtered_list))
    
    split_index = int(len(filtered_list)*(1 - test_size))
    train_split = filtered_list[:split_index]
    test_split = filtered_list[split_index:]
    # print(f'num train: ', len(train_split), ' num test: ', len(test_split),'\n')
    
    if fp_out is None:
        fp_out = 'sql_data'
    if not os.path.exists(fp_out):
        os.makedirs(fp_out)
        
    train_file = fp_out + 'train.txt'
    format_n_save(train_split, train_file)

    test_file = fp_out + 'test.txt'
    format_n_save(test_split, test_file)

    print(f"\nData processing complete. Files saved in '{fp_out}' directory.")


if __name__ == '__main__':
    process_and_save_sql_data()
