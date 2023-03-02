#pragma once

#include <CoreBgs.hpp>

#include <map>
#include <string>
#include <iostream>

namespace sky360lib::bgs
{
    class CoreParameters
    {
    public:
        void setBgs(CoreBgs* _bgs)
        {
            m_coreBgs = _bgs;
        }

        friend class CoreBgs;

    protected:

        CoreBgs* m_coreBgs;
    };
}