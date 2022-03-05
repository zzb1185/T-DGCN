import input_data as id


def load_data(data_name):
    if data_name == 'sz':
        data, adj = id.load_sz_data('sz')
    if data_name == 'los':
        data, adj = id.load_los_data('los')
    if data_name == '533':
        data, adj = id.load_cql_data('533')
    if data_name == '1141':
        data, adj = id.load_1141_data('1141')
    if data_name == '504':
        data, adj = id.load_504_data('504')
    if data_name == '504双链接DTW回归':
        data, adj = id.load_504SLJDTWHG_data('504')
    if data_name == '432':
        data, adj = id.load_432_data('432')
    if data_name == '432_350':
        data, adj = id.load_432350_data('432_350')
    if data_name == '432_350_1':
        data, adj = id.load_432350duizhao_data('432_350')
    if data_name =='63bwd':
        data, adj = id.load_63bwd_data('63bwd')

    return data,adj
