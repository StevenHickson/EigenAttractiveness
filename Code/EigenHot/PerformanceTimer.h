#pragma once
#include <Windows.h>

//class for high precision timing. Accurate to less than 10 microseconds
//to be safe, accurate at a minimum of 100 microseconds. Do not use for timing greater than 100 ms, use StopWatch class instead.
class PerformanceTimer
{
public:
	// Constructor.
	PerformanceTimer()
	{
		bStarted = false;
		// Ticks per second.
		// This is also a classic problem in Windows that may have timing error if Performance Clocking is allowed such as intel SpeedStep, TurboBoost, etc.
		QueryPerformanceFrequency( &liFreq );
	}

	// Start counter.
	void Start()
	{
		liStart.QuadPart = 0;
		bStarted = true;
		QueryPerformanceCounter( &liStart );
	}

	// Stop counter.
	void Stop()
	{
		liEnd.QuadPart = 0;
		QueryPerformanceCounter( &liEnd );
		bStarted = false;
	}

	// Get duration in seconds
	long double Duration()
	{
		//make sure you stop the timer first, if it isn't stopped then it stops automatically and resumes where it left off
		if(bStarted) {
			Stop();
			LARGE_INTEGER liTmp = liStart;
			long double ret = ( liEnd.QuadPart - liStart.QuadPart) /
				long double( liFreq.QuadPart );
			Start();
			liStart = liTmp;
			return ret;
		} else
			return ( liEnd.QuadPart - liStart.QuadPart) /
			long double( liFreq.QuadPart );
	}

private:
	LARGE_INTEGER liStart;
	LARGE_INTEGER liEnd;
	LARGE_INTEGER liFreq;
	bool bStarted;
};
