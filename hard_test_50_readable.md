# Hard Test Problems (50 Problems, IMO-Level Difficulty)

## Number Theory

### nt_01
**Question:** Let $N = 2^4 \cdot 3^2 \cdot 5 \cdot 7 \cdot 11 \cdot 13 = 720720$. Let $\mu$ denote the Mobius function, $\sigma$ the sum-of-divisors function, and define $F(N) = \sum_{d \mid N} \mu(d) \cdot \sigma(N/d)$. Compute $F(N) \bmod 10^5$.

**Answer:** 38240

---

### nt_02
**Question:** Let $k$ be the largest non-negative integer such that $3^k$ divides $\binom{300}{100}$. Find $k$.

**Answer:** 3

---

### nt_03
**Question:** Compute $\displaystyle\sum_{i=1}^{10000} \gcd(i, 10000)$. Find the answer modulo $10^5$.

**Answer:** 26000

---

### nt_04
**Question:** How many square-free integers $n$ with $1 \le n \le 50000$ have the property that the sum of the digits of $n$ (in base 10) is divisible by $7$?

**Answer:** 4920

---

### nt_05
**Question:** Find the smallest positive integer $x$ such that $x \equiv 3 \pmod{7}$, $x \equiv 5 \pmod{11}$, $x \equiv 8 \pmod{13}$, $x \equiv 2 \pmod{17}$, $x \equiv 10 \pmod{19}$, and $x \equiv 1 \pmod{23}$. Find $x \bmod 10^5$.

**Answer:** 9349

---

### nt_06
**Question:** Let $p = 99991$, which is prime. How many distinct quadratic residues modulo $p$ are there? (That is, how many distinct values does $a^2 \bmod p$ take for $a = 1, 2, \ldots, p-1$?)

**Answer:** 49995

---

### nt_07
**Question:** A positive integer $n$ is called \emph{powerful} if for every prime $p$ dividing $n$, we also have $p^2 \mid n$. (For example, $1, 4, 8, 9, 16, 25, 27, 32, 36, \ldots$ are powerful.) How many powerful numbers are there in the range $1 \le n \le 50000$?

**Answer:** 434

---

### nt_08
**Question:** Find the sum of all primitive roots modulo $59$. Give your answer modulo $10^5$.

**Answer:** 886

---

### nt_09
**Question:** Write $10^8$ in the \emph{factorial number system} (also known as factoriadic): $10^8 = d_k \cdot k! + d_{k-1} \cdot (k-1)! + \cdots + d_2 \cdot 2! + d_1 \cdot 1!$, where $0 \le d_i \le i$ for each $i$. Compute the digit sum $d_1 + d_2 + \cdots + d_k$.

**Answer:** 29

---

### nt_10
**Question:** Let $\sigma(n)$ denote the sum of all positive divisors of $n$. Compute $\sigma(10!) \bmod 10^5$, where $10! = 3628800$.

**Answer:** 34088

---

### nt_11
**Question:** How many positive integers $n$ with $1 \le n \le 100000$ can be expressed as the sum of two squares of non-negative integers? (That is, $n = a^2 + b^2$ for some integers $a, b \ge 0$.)

**Answer:** 24028

---

### nt_12
**Question:** The Farey sequence $F_n$ is the sequence of completely reduced fractions $\frac{a}{b}$ with $0 \le \frac{a}{b} \le 1$ and $b \le n$, arranged in increasing order. Find the number of terms $|F_{100}|$ in the Farey sequence $F_{100}$. Give your answer modulo $10^5$.

**Answer:** 3045

---

### nt_13
**Question:** Let $q(n)$ denote the number of partitions of $n$ into distinct parts. For example, $q(5) = 3$ since $5 = 4+1 = 3+2$. Compute $q(100) \bmod 10^5$.

**Answer:** 44793

---

## Algebra

### alg_01
**Question:** Let $r_1, r_2, r_3, r_4, r_5$ be the roots (in $\mathbb{C}$) of the polynomial $P(x) = x^5 - 3x^4 + 5x^3 - 7x^2 + 11x - 13$. Compute $r_1^3 + r_2^3 + r_3^3 + r_4^3 + r_5^3$. If the answer is negative, give the answer modulo $10^5$ (i.e., add $10^5$).

**Answer:** 3

---

### alg_02
**Question:** Let $F_n$ denote the $n$-th Fibonacci number with $F_1 = F_2 = 1$. Compute $F_{10^7} \bmod 99991$.

**Answer:** 29526

---

### alg_03
**Question:** Define a sequence by $a_0 = 1$, $a_1 = 1$, and $a_n = 3a_{n-1} - a_{n-2}$ for $n \ge 2$. Compute $a_{50} \bmod 10^5$.

**Answer:** 69026

---

### alg_04
**Question:** Compute the resultant $\text{Res}(f, g)$ of the polynomials $f(x) = x^3 - 2x + 1$ and $g(x) = x^2 - 3$. If the answer is negative, give the answer modulo $10^5$.

**Answer:** 99998

---

### alg_05
**Question:** Let $T_n(x)$ denote the $n$-th Chebyshev polynomial of the first kind, defined by $T_0(x) = 1$, $T_1(x) = x$, and $T_n(x) = 2x \cdot T_{n-1}(x) - T_{n-2}(x)$. Compute $T_{100}(3) \bmod 10^5$.

**Answer:** 71937

---

### alg_06
**Question:** Let $M$ be the $5 \times 5$ matrix with entry $M_{ij} = (ij \bmod 5) + 1$ for $0 \le i, j \le 4$. Compute the permanent of $M$, defined as $\text{perm}(M) = \sum_{\sigma \in S_5} \prod_{i=0}^{4} M_{i,\sigma(i)}$.

**Answer:** 8114

---

### alg_07
**Question:** Let $r_1, r_2, r_3, r_4$ be the four roots of $x^4 + x^3 + x^2 + x + 1 = 0$. Compute $r_1^{100} + r_2^{100} + r_3^{100} + r_4^{100}$.

**Answer:** 4

---

### alg_08
**Question:** Compute $\displaystyle\sum_{k=1}^{10^6} \lfloor \sqrt{k} \rfloor$. Give your answer modulo $10^5$.

**Answer:** 67500

---

### alg_09
**Question:** Evaluate the continued fraction $[1; 2, 3, 4, 5, 6, 7, 8, 9, 10]$. Express the result as $\frac{p}{q}$ in lowest terms. Find $(p + q) \bmod 10^5$.

**Answer:** 14721

---

### alg_10
**Question:** Let $A = \begin{pmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 0 \end{pmatrix}$. Compute $\text{tr}(A^{10}) \bmod 10^5$, where $\text{tr}$ denotes the trace.

**Answer:** 23300

---

### alg_11
**Question:** Compute $\displaystyle\sum_{k=1}^{99999} \left\lfloor \frac{99999}{k} \right\rfloor$. Give your answer modulo $10^5$.

**Answer:** 66714

---

### alg_12
**Question:** Let $V$ be the $6 \times 6$ Vandermonde matrix with nodes $1, 2, 3, 4, 5, 6$, i.e., $V_{ij} = i^{j-1}$ for $1 \le i, j \le 6$. Compute $\det(V) \bmod 10^5$.

**Answer:** 34560

---

### alg_13
**Question:** The Collatz function is defined as $f(n) = n/2$ if $n$ is even, and $f(n) = 3n+1$ if $n$ is odd. Starting from $n = 27$, how many applications of $f$ are needed to first reach $1$?

**Answer:** 111

---

## Combinatorics

### comb_01
**Question:** A derangement is a permutation with no fixed points. Let $D_n$ denote the number of derangements of $\{1, 2, \ldots, n\}$. Compute $D_{20} \bmod 10^5$.

**Answer:** 2121

---

### comb_02
**Question:** The $n$-th Catalan number is $C_n = \frac{1}{n+1}\binom{2n}{n}$. Compute $C_{30} \bmod 10^5$.

**Answer:** 92304

---

### comb_03
**Question:** The Stirling number of the second kind $S(n,k)$ counts the number of ways to partition a set of $n$ elements into exactly $k$ non-empty subsets. Compute $S(20, 10) \bmod 10^5$.

**Answer:** 64655

---

### comb_04
**Question:** In an election, candidate A receives $100$ votes and candidate B receives $80$ votes, cast sequentially. In how many orderings of the votes is A strictly ahead of B throughout the entire counting process? Give your answer modulo $10^5$.

**Answer:** 35790

---

### comb_05
**Question:** A \emph{reduced} (or \emph{normalized}) Latin square of order $n$ is a Latin square where the first row and first column are both in natural order $1, 2, \ldots, n$. How many reduced Latin squares of order $5$ are there?

**Answer:** 56

---

### comb_06
**Question:** How many binary strings of length $30$ contain no three consecutive $1$s?

**Answer:** 50096

---

### comb_07
**Question:** How many distinct necklaces can be made with $13$ beads, where each bead is one of $3$ colors? Two necklaces are the same if one can be obtained from the other by rotation. (Reflections are considered different.) Give your answer modulo $10^5$.

**Answer:** 22643

---

### comb_08
**Question:** Let $p(n)$ denote the number of (unrestricted) partitions of $n$ into positive integer parts. For example, $p(4) = 5$ since $4 = 3+1 = 2+2 = 2+1+1 = 1+1+1+1$. Compute $p(50) \bmod 10^5$.

**Answer:** 4226

---

### comb_09
**Question:** How many surjective (onto) functions are there from a set of $15$ elements to a set of $7$ elements? Give your answer modulo $10^5$.

**Answer:** 18320

---

### comb_10
**Question:** How many lattice paths are there from $(0,0)$ to $(15,15)$ using unit steps right $(1,0)$ and up $(0,1)$, such that the path never goes strictly above the diagonal $y = x$? Give your answer modulo $10^5$.

**Answer:** 94845

---

### comb_11
**Question:** The complete bipartite graph $K_{3,3}$ has $6$ vertices and $9$ edges. A proper coloring assigns a color from $\{1, 2, 3, 4, 5\}$ to each vertex such that no two adjacent vertices share a color. How many proper colorings of $K_{3,3}$ with $5$ colors are there?

**Answer:** 2420

---

### comb_12
**Question:** By Cayley's formula, the number of labeled trees on $n$ vertices is $n^{n-2}$. Compute the number of labeled trees on $15$ vertices, modulo $10^5$.

**Answer:** 59375

---

## Geometry

### geo_01
**Question:** How many lattice points $(x, y)$ with integer coordinates satisfy $x^2 + y^2 \le 100^2$?

**Answer:** 31417

---

### geo_02
**Question:** A triangle has vertices at $(0,0)$, $(123, 456)$, and $(789, 12)$. How many lattice points (points with integer coordinates) lie on or inside this triangle? Give your answer modulo $10^5$.

**Answer:** 79269

---

### geo_03
**Question:** A triangle has side lengths $a = 2023$, $b = 2024$, and $c = 2025$. Compute $\lfloor A \rfloor$, where $A$ is the area of the triangle. Give your answer modulo $10^5$.

**Answer:** 73868

---

### geo_04
**Question:** A regular $17$-gon is inscribed in a circle of radius $1$. Let $S$ be the sum of the distances from one vertex to all other $16$ vertices. Compute $\lfloor 10000 \cdot S \rfloor$.

**Answer:** 15834

---

### geo_05
**Question:** How many Pythagorean triples $(a, b, c)$ of positive integers satisfy $a \le b < c \le 1000$ and $a^2 + b^2 = c^2$?

**Answer:** 881

---

### geo_06
**Question:** Two circles, each of radius $5$, have centers at $(0,0)$ and $(3,4)$. Let $A$ be the area of their intersection. Compute $\lfloor 100 \cdot A \rfloor$.

**Answer:** 3070

---

### geo_07
**Question:** Triangle $ABC$ has vertices $A = (0,0)$, $B = (8,0)$, $C = (3,7)$. Let $O$ be the circumcenter and $G$ the centroid. Compute $\lfloor 1000 \cdot |OG| \rfloor$.

**Answer:** 346

---

### geo_08
**Question:** A convex polygon has vertices $(1,1)$, $(5,2)$, $(7,6)$, $(4,8)$, $(0,5)$ listed in order. Compute its area.

**Answer:** 30

---

### geo_09
**Question:** A triangle has side lengths $a = 13$, $b = 14$, $c = 15$. Let $r$ be its inradius and $r_a$ the exradius opposite side $a$. Compute $\lfloor 1000(r_a - r) \rfloor$.

**Answer:** 6500

---

### geo_10
**Question:** How many triangles with integer side lengths have perimeter exactly $100$? (Triangles that differ only in the labeling of sides are considered the same.)

**Answer:** 208

---

### geo_11
**Question:** A circle has center $O = (0,0)$ and radius $10$. The point $P = (15, 0)$ lies outside the circle. A line through $P$ intersects the circle at points $A$ and $B$. Compute the product $PA \cdot PB$. (This value is independent of the choice of line.)

**Answer:** 125

---

### geo_12
**Question:** How many lattice points $(x, y)$ with integer coordinates lie strictly between (not including the endpoints) $(0, 0)$ and $(2024, 1230)$ on the line segment connecting them?

**Answer:** 1

---


**Total: 50 problems**
