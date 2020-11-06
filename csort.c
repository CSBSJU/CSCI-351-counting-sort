/* assert */
#include <assert.h>

/* OpenMP API */
#include <omp.h>

/* EXIT_SUCCESS, rand */
#include <stdlib.h>

/* strtol */
#include <stdio.h>

static inline int
MKPWROF2(int v)
{
  v--;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  return v+1;
}

static void
scan(unsigned const n, unsigned * const a)
{
  unsigned * b;

  #pragma omp parallel
  {
    /* get number of openmp threads as a power of 2 */
    int const size = MKPWROF2(omp_get_num_threads());
    /* get thread rank */
    int const rank = omp_get_thread_num();

    if (0 == rank) {
      /* allocate memory for partial sums */
      b = calloc(size, sizeof(*b));
      assert(b);
    }

    /* local up-sweep (summation) */
    unsigned sum = 0;
    #pragma omp for schedule(static)
    for (unsigned i = 0; i < n; ++i)
      sum += a[i];

    /* record private copy of b */
    unsigned * const c = b;

    /* record local up-sweep */
    c[rank] = sum;

    /* global up-sweep (reduction) */
    int off=1;
    for (int d = size / 2; d > 0; d /= 2, off *= 2) {
      #pragma omp barrier
      if (rank < d) {
        unsigned const ai = off * (2 * rank + 1) - 1;
        unsigned const bi = off * (2 * rank + 2) - 1;
        c[bi] += c[ai];
      }
    }

    /* clear for global down-sweep */
    if (0 == rank)
      c[size - 1] = 0;

    /* global down-sweep */
    for (int d = 1; d < size; d *= 2) {
      off /= 2;
      #pragma omp barrier
      if (rank < d) {
        unsigned const ai  = off * (2 * rank + 1) - 1;
        unsigned const bi  = off * (2 * rank + 2) - 1;
        unsigned const t = c[ai];
        c[ai] = c[bi];
        c[bi] += t;
      }
    }
    #pragma omp barrier

    /* local scan using offset from global scan */
    unsigned p=c[rank];
    #pragma omp for schedule(static) nowait
    for (unsigned i = 0; i < n; ++i) {
      unsigned const t = a[i];
      a[i] = p;
      p += t;
    }
  }

  /* release memory for partial sums */
  free(b);
}

static int
csort(unsigned const k,
      unsigned const n,
      unsigned const * const in,
      unsigned       * const out)
{
  unsigned t;

# pragma omp parallel master
  t = omp_get_num_threads();

  unsigned * const count = calloc(t * k + 1, sizeof(*count));
  if (NULL == count) {
    return -1;
  }

  double const ts1 = omp_get_wtime();
# pragma omp parallel for
  for (unsigned i = 0; i < n; i++) {
    count[in[i] * omp_get_num_threads() + omp_get_thread_num()]++;
  }
  double const te1 = omp_get_wtime();
  printf("timer1: %lf\n", te1 - ts1);

  double const ts2 = omp_get_wtime();
#if 1
 unsigned total = 0;
  for (unsigned i = 0; i <= t * k; i++) {
    unsigned const counti = count[i];
    count[i] = total;
    total += counti;
  }
#elif 0
  scan(t * k, count);
#else
  unsigned total = 0;
# pragma omp simd reduction (inscan, +:total)
  for (unsigned i = 0; i <= t * k; i++) {
    count[i] = total;
#   pragma omp scan exclusive(total)
    total += count[i];
  }
#endif
  double const te2 = omp_get_wtime();
  printf("timer2: %lf\n", te2 - ts2);

  double const ts3 = omp_get_wtime();
# pragma omp parallel for
  for (unsigned i = 0; i < n; i++) {
    out[count[in[i] * omp_get_num_threads() + omp_get_thread_num()]++] = in[i];
  }
  double const te3 = omp_get_wtime();
  printf("timer3: %lf\n", te3 - ts3);

  free(count);

  return 0;
}

int
main(int argc, char *argv[]) {
  /* Get array size from command line */
  unsigned n = strtol(argv[1], NULL, 10);

  /* Get key size from command line */
  unsigned k = strtol(argv[2], NULL, 10);

  /* Allocate memory */
  unsigned * const a = malloc(n * sizeof(*a));
  unsigned * const b = malloc(n * sizeof(*b));

  /* Populate with random values */
  for (unsigned i = 0; i < n; i++) {
    a[i] = rand() % (1u << k);
  }

  /* Sort array */
  int const ret = csort(1u << k, n, a, b);
  assert(0 == ret);

  /* Validate sorted array */
  for (unsigned i = 1; i < n; i++) {
    assert(b[i] >= b[i - 1]);
  }

  /* Free memory */
  free(a);
  free(b);

  return EXIT_SUCCESS;
}
