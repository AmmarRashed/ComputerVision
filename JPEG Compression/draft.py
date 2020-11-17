def matricize(n, nrows):
    ncols = n//nrows
    flag = False
    for i in range(n):
        row = i//ncols
        col = i%ncols
        end = '\n' if col == ncols-1 else ''

        print(row, end=end)

matricize(10, 2)