import input_data as id


def load_data(data_name):
    if data_name == 'test_tdgcn':
        data, adj = id.load_testtdgcn_data('test')
    if data_name == 'test_tgcn':
        data, adj = id.load_testtgcn_data('test')
    # if data_name == 'sz':
    #     data, adj = id.load_sz_data('sz')
    # if data_name == 'los':
    #     data, adj = id.load_los_data('los')
    # if data_name == '432_350':
    #     data, adj = id.load_432350_data('432_350')
    # if data_name == '432_350_1':
    #     data, adj = id.load_432350duizhao_data('432_350')


    return data,adj
