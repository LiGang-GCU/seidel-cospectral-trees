"""Recompute alkane-skeleton counts n=1..12 and total the census range."""
import networkx as nx

# Orders 13..24 as reported in tab:chem (this paper's own census).
TAB = {
    13: 802, 14: 1858, 15: 4347, 16: 10359, 17: 24894, 18: 60523,
    19: 148284, 20: 366319, 21: 910726, 22: 2278658, 23: 5731580,
    24: 14490245,
}

small = {1: 1, 2: 1}
for n in range(3, 13):
    small[n] = sum(1 for T in nx.nonisomorphic_trees(n)
          if max(d for _, d in T.degree()) <= 4)

for n in sorted(small):
    print("n=%2d  %d" % (n, small[n]))

allc = dict(small)
allc.update(TAB)
tot_1_24 = sum(allc[n] for n in range(1, 25))
tot_4_24 = sum(allc[n] for n in range(4, 25))
print()
print("sum n=1..24 :", format(tot_1_24, ","))
print("sum n=4..24 :", format(tot_4_24, ","))
print("sum n=13..24:", format(sum(TAB.values()), ","))
print("paper states: 24,029,256")
