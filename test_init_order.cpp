#include <iostream>
#include <cstdio>

class TestInit {
public:
    TestInit() {
        // Use C stdio which doesn't require static initialization
        fprintf(stderr, "TestInit constructor called\n");
        fflush(stderr);
        
        // Now try iostream
        std::cerr << "iostream test in constructor" << std::endl;
    }
};

// Global object to test initialization order
TestInit global_test_obj;

extern "C" {
    void test_iostream_init() {
        fprintf(stderr, "test_iostream_init called\n");
        fflush(stderr);
        std::cout << "iostream test in function" << std::endl;
    }
}