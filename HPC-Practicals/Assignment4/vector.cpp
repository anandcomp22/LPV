#include <iostream>
#include <omp.h>
using namespace std;

int main() {
    int n;

    cout << "Enter size of vectors: ";
    cin >> n;

    int A[n], B[n], C[n];

    cout << "Enter elements of vector A:\n";
    for (int i = 0; i < n; i++) cin >> A[i];

    cout << "Enter elements of vector B:\n";
    for (int i = 0; i < n; i++) cin >> B[i];

    double start = omp_get_wtime();

    // Parallel vector addition
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        C[i] = A[i] + B[i];
    }

    double end = omp_get_wtime();

    cout << "Result vector:\n";
    for (int i = 0; i < n; i++) cout << C[i] << " ";

    cout << "\nExecution Time: " << (end - start) << " seconds\n";

    return 0;
}