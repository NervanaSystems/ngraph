#pragma once

#include <algorithm>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <vector>

#include "util.hpp"

namespace ngraph
{
    template <typename T>
    class tree;
    using scalar_tree = ngraph::tree<size_t>;
}

//================================================================================================
//
//================================================================================================
template <typename T>
class ngraph::tree
{
public:
    tree(T s)
        : m_list{}
        , m_value{s}
        , m_is_list{false}
    {
    }

    tree(const std::initializer_list<tree<T>>& list)
        : m_list{}
        , m_value{0}
        , m_is_list{true}
    {
        m_list = list;
    }

    tree(const std::vector<T>& list)
        : m_list{}
        , m_value{0}
        , m_is_list{true}
    {
        for (auto s : list)
        {
            m_list.push_back(tree(s));
        }
    }

    bool                     is_list() const { return m_is_list; }
    T                        get_value() const { return m_value; }
    const std::vector<tree>& get_list() const { return m_list; }
    static void traverse_tree(tree& s, std::function<void(T*)> func)
    {
        if (s.is_list())
        {
            for (tree& s1 : s.m_list)
            {
                traverse_tree(s1, func);
            }
        }
        else
        {
            func(&(s.m_value));
        }
    }

    friend std::ostream& operator<<(std::ostream& out, const tree& s)
    {
        if (s.is_list())
        {
            out << "(" << join(s.get_list(), ", ") << ")";
        }
        else
        {
            out << s.get_value();
        }
        return out;
    }

    T reduce(const std::function<T(T, T)>& func) const
    {
        size_t rc;
        if (is_list())
        {
            switch (m_list.size())
            {
            case 0: rc = 0; break;
            case 1: rc = m_list[0].reduce(func); break;
            default:
                rc = m_list[0].reduce(func);
                for (int i = 1; i < m_list.size(); i++)
                {
                    rc = func(rc, m_list[i].reduce(func));
                }
                break;
            }
        }
        else
        {
            rc = m_value;
        }
        return rc;
    }

private:
    std::vector<tree> m_list;
    T                 m_value;
    bool              m_is_list;
};
