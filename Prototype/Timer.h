//
// Created by Esha Uboweja on 12/6/16.
//

#pragma once

#include <chrono>

namespace Utility
{
    typedef std::chrono::high_resolution_clock clock;
    typedef std::chrono::milliseconds milliseconds;
    typedef std::chrono::microseconds microseconds;

    clock::time_point now()
    {
        return clock::now();
    }

    milliseconds intervalMs(const clock::time_point &startT,
                            const clock::time_point &endT)
    {
        return std::chrono::duration_cast<milliseconds>(endT - startT);
    }

    microseconds intervalUs(const clock::time_point &startT,
                            const clock::time_point &endT)
    {
        return std::chrono::duration_cast<microseconds>(endT - startT);
    }

    class Timer
    {
    public:
        Timer(): mStartT(now()) {}

        clock::time_point restart()
        {
            mStartT = now();
            return mStartT;
        }

        milliseconds elapsedMs()
        {
            return intervalMs(mStartT, now());
        }

        microseconds elapsedUs()
        {
            return intervalUs(mStartT, now());
        }

    private:
        clock::time_point mStartT;
    };
}
