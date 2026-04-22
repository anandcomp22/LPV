#include <iostream>
#include <queue>
#include <omp.h>
using namespace std;

class Node {
public:
    int data;
    Node *left, *right;
};

// Insert node in binary tree
Node* insert(Node* root, int data) {
    if (!root) {
        root = new Node();
        root->data = data;
        root->left = root->right = NULL;
        return root;
    }

    queue<Node*> q;
    q.push(root);

    while (!q.empty()) {
        Node* temp = q.front();
        q.pop();

        if (!temp->left) {
            temp->left = new Node();
            temp->left->data = data;
            temp->left->left = temp->left->right = NULL;
            return root;
        } else {
            q.push(temp->left);
        }

        if (!temp->right) {
            temp->right = new Node();
            temp->right->data = data;
            temp->right->left = temp->right->right = NULL;
            return root;
        } else {
            q.push(temp->right);
        }
    }
    return root;
}

// Parallel BFS
void parallelBFS(Node* root) {
    queue<Node*> q;
    q.push(root);

    while (!q.empty()) {
        int size = q.size();

        #pragma omp parallel for
        for (int i = 0; i < size; i++) {
            Node* curr;

            #pragma omp critical
            {
                curr = q.front();
                q.pop();
                cout << curr->data << " ";
            }

            #pragma omp critical
            {
                if (curr->left) q.push(curr->left);
                if (curr->right) q.push(curr->right);
            }
        }
    }
}

int main() {
    Node* root = NULL;
    int data;
    char ch;

    do {
        cout << "Enter data: ";
        cin >> data;
        root = insert(root, data);

        cout << "Add more? (y/n): ";
        cin >> ch;
    } while (ch == 'y' || ch == 'Y');

    cout << "BFS Traversal: ";
    parallelBFS(root);

    return 0;
}