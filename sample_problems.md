# AIMO3 Reference Problems (10 Sample Problems)

## Problem 1 (ID: 42d360) - CORRECT
**Topic:** Number Theory / Base Representation

Ken starts with a positive integer n on a blackboard. Each move, he picks a base b (where 2 <= b <= current number m), writes m in base b, then replaces m with the sum of its base-b digits. He repeats until he reaches 1.

For all starting values n from 1 to 10^(10^5), what is the largest possible number of moves M? Find M mod 10^5.

**Answer:** 32193
**Time:** 64s | Python Calls: 31

---

## Problem 2 (ID: 0e644e) - CORRECT
**Topic:** Geometry

Triangle ABC has integer side lengths with AB < AC. Points D and E are on segments BC and AC such that AD = AE = AB. Line DE meets AB at X. The circles through B,X,D and through C,E,D intersect again at Y (not D). Y lies on line AD. Find the unique triangle with minimal perimeter having this property. If the side lengths are a = BC, b = CA, c = AB, find (a * b * c) mod 10^5.

**Answer:** 336
**Time:** 133s | Python Calls: 54

---

## Problem 3 (ID: 92ba6a) - CORRECT
**Topic:** Algebra / Systems of Equations

Alice and Bob each hold some whole number of sweets. Alice says: "If we each add our number of sweets to our positive integer age, my total would be double yours. If we multiply instead, my result would be four times yours." Bob replies: "Give me 5 of your sweets, and then both our sum and product would be equal." What is the product of Alice and Bob's ages?

**Answer:** 50
**Time:** 19s | Python Calls: 4

---

## Problem 4 (ID: a295e9) - CORRECT
**Topic:** Combinatorics / Tiling

A 500 x 500 square is divided into k rectangles, each with integer side lengths. No two rectangles have the same perimeter. What is the largest possible k? Find k mod 10^5.

**Answer:** 520
**Time:** 251s | Python Calls: 48

---

## Problem 5 (ID: 26de63) - CORRECT
**Topic:** Number Theory / Floor Functions

Define f(n) = sum over i=1..n, sum over j=1..n of j^1024 * floor(1/j + (n-i)/n). Let M = 2*3*5*7*11*13 and N = f(M^15) - f(M^15 - 1). Let k be the largest non-negative integer such that 2^k divides N. Find 2^k mod 5^7.

**Answer:** 32951
**Time:** 53s | Python Calls: 10

---

## Problem 6 (ID: dd7f5e) - CORRECT
**Topic:** Algebra / Polynomial / Convolution

Consider functions from integers to integers where only finitely many values are nonzero. Define a "product" of two such functions as the sum of their pointwise products, and a shift operator that shifts the function by n positions.

A function alpha is "shifty" if: (1) alpha(m) = 0 for m < 0 and m > 8, and (2) there exists another function beta and two distinct integers k, l such that the shifted inner product of alpha with beta equals 1 at positions k and l, and 0 everywhere else.

How many shifty functions exist?

**Answer:** 160
**Time:** 206s | Python Calls: 75

---

## Problem 7 (ID: 86e8e5) - WRONG
**Topic:** Number Theory / Divisors / Asymptotics

A positive integer is called "n-Norwegian" if it has three distinct positive divisors that sum to n. f(n) is the smallest n-Norwegian number. Let M = 3^(2025!). Define g(c) = floor(2025! * f(M+c) / M) / 2025!.

Compute g(0) + g(4M) + g(1848374) + g(10162574) + g(265710644) + g(44636594) = p/q where p,q are coprime. Find (p+q) mod 99991.

**Model Answer:** 96985 (WRONG)
**Correct Answer:** 8687
**Time:** ~600s | Python Calls: 71 | Votes: 1 (no consensus)

---

## Problem 8 (ID: 424e18) - CORRECT
**Topic:** Combinatorics / Tournament

A tournament has 2^20 runners, each with a different speed (faster always wins). There are 20 rounds. Each round, runners are paired so both in each pair have the same current score. The winner of a race in round i gets 2^(20-i) points. After all rounds, runners are ranked by score. How many distinct final orderings N are possible? Find the largest k such that 10^k divides N, then find k mod 10^5.

**Answer:** 21818
**Time:** 150s | Python Calls: 38

---

## Problem 9 (ID: 641659) - CORRECT
**Topic:** Geometry / Fibonacci

Triangle ABC with AB != AC has circumcircle Omega and incircle omega touching BC, CA, AB at D, E, F. The circumcircle of AFE meets Omega at K, and K' is the reflection of K over EF. N is the foot of the perpendicular from D to EF. A circle tangent to line BN passing through B and K meets BC again at T.

A triangle is "n-tastic" if BD = F_n (nth Fibonacci number), CD = F_{n+1}, and KNKB is cyclic. For each n-tastic triangle, a_n is the max of (CT * NB)/(BT * NE). Let alpha be the smallest real number bounding a_{2n} for large n. Given alpha = p + sqrt(q), find floor(p^(q^p)) mod 99991.

**Answer:** 57447
**Time:** 260s | Python Calls: 69

---

## Problem 10 (ID: 9c1c5f) - CORRECT
**Topic:** Functional Equations

Find a function f from positive integers to positive integers such that f(m) + f(n) = f(m + n + mn) for all positive integers m, n. Given that f(n) <= 1000 for all n <= 1000, how many different values can f(2024) take?

**Answer:** 580
**Time:** 88s | Python Calls: 38

---

## Summary
- **Score:** 9/10 (90%)
- **Total Time:** ~1864s across all problems
- **Average Python Calls:** 43.8 per problem
- **Failed Problem:** #7 (Norwegian numbers) - required asymptotic analysis of divisor sums for astronomically large numbers
