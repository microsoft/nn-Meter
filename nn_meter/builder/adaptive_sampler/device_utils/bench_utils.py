def fetech_tf_bench_results(result_str):
        #print('result:',result_str)
        if rfind_assign_int(result_str, 'count') >= 2:
            std_ms = rfind_assign_float(result_str, 'std') / 1e3
            avg_ms = rfind_assign_float(result_str, 'avg') / 1e3
        else:
            std_ms = 0
            avg_ms = rfind_assign_float(result_str, 'curr') / 1e3

        return std_ms,avg_ms
def rfind_assign(s, mark):
    mark += "="
    p = s.rfind(mark)
    assert p != -1
    l_idx = p + len(mark)
    r_idx = l_idx
    while s[r_idx] not in [' ', '\n']:
        r_idx += 1
    return s[l_idx: r_idx]


def rfind_assign_float(s, mark):
    return float(rfind_assign(s, mark))


def rfind_assign_int(s, mark):
    return int(rfind_assign(s, mark))


def table_try_float(table):
    for i in range(len(table)):
        for j in range(len(table[i])):
            try:
                table[i][j] = float(table[i][j])
            except:
                pass
    return table
