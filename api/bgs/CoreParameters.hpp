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
        CoreParameters() {}
        CoreParameters(const CoreParameters &) {};
        virtual ~CoreParameters() {};

        template<class _T>
        void set(int _param, _T value)
        {
            if (m_paramsMap.contains(_param))
            {
                *((_T*)m_paramsMap.at(_param).valueP) = value;
                
                paramUpdated(_param);

                if (m_paramsMap.at(_param).requiresRestart && m_coreBgs != nullptr)
                {
                    m_coreBgs->restart();
                }
            }
            else
            {
                std::cerr << "Trying to set unknow parameter: " << _param << std::endl;
            }
        }

        template<class _T>
        _T get(int _param) const
        {
            return *(_T*)m_paramsMap.at(_param).valueP;
        }

        int32_t getInt32(int _param) const { return get<int32_t>(_param); }
        uint32_t getUInt32(int _param) const { return get<uint32_t>(_param); }
        int64_t getInt64(int _param) const { return get<int64_t>(_param); }
        uint64_t getUInt64(int _param) const { return get<uint64_t>(_param); }
        float getFloat(int _param) const { return get<float>(_param); }
        double getDouble(int _param) const { return get<double>(_param); }

        void setBgs(CoreBgs* _bgs)
        {
            m_coreBgs = _bgs;
        }

        friend class CoreBgs;

    protected:
        virtual void paramUpdated(int _param) = 0;

        struct ParamMap
        {
            ParamMap(std::string _name, bool _requiresRestart, void* _valueP)
                : name(_name), requiresRestart(_requiresRestart), valueP(_valueP)
            {}

            ParamMap(const ParamMap& _param)
                : name(_param.name), requiresRestart(_param.requiresRestart), valueP(_param.valueP)
            {}
            
            std::string name;
            bool requiresRestart;
            void* valueP;
        };

        std::map<int, ParamMap> m_paramsMap;
        CoreBgs* m_coreBgs;
    };
}