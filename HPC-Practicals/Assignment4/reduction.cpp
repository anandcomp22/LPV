#include <iostream>
#include <omp.h>
using namespace std;

int main() {
    int n;

    cout << "Enter number of elements: ";
    cin >> n;

    int arr[n];
    cout << "Enter elements:\n";
    for (int i = 0; i < n; i++) cin >> arr[i];

    int sum = 0;
    int min_val = arr[0];
    int max_val = arr[0];
    double start = omp_get_wtime();
    // Sum
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < n; i++) {
        sum += arr[i];
    }
    // Min
    #pragma omp parallel for reduction(min:min_val)
    for (int i = 0; i < n; i++) {
        if (arr[i] < min_val)
            min_val = arr[i];
    }
    // Max
    #pragma omp parallel for reduction(max:max_val)
    for (int i = 0; i < n; i++) {
        if (arr[i] > max_val)
            max_val = arr[i];
    }
    double end = omp_get_wtime();
    double avg = (double)sum / n; 
    cout << "Sum: " << sum << endl;
    cout << "Minimum: " << min_val << endl;
    cout << "Maximum: " << max_val << endl;
    cout << "Average: " << avg << endl;
    cout << "Execution Time: " << (end - start) << " seconds\n";
    return 0;
}