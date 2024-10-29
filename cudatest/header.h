#pragma once

#include <iostream>
#include <type_traits>

void print_matrix(double **C, int N);

template <typename test_type, typename... Ts>
constexpr bool are_all_same_v =
    std::conjunction_v<std::is_same<test_type, Ts>...>::value;

void test_arr_form();
