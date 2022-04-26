#pragma once

#include <unordered_map>
#include <string>
#include <string>
#include <chrono>


class TimeLogger
{
private:
	TimeLogger();
	TimeLogger(const TimeLogger&) = delete;
	TimeLogger& operator=(TimeLogger&) = delete;
	static TimeLogger* _instance;

	using TimeDiff_t = std::pair<std::chrono::time_point<std::chrono::system_clock>, std::chrono::time_point<std::chrono::system_clock>>;

	static std::unordered_map<std::string, TimeDiff_t> _timeMeasure;
public:
	static TimeLogger* getInstance();

	static void StartMeasure(const std::string& label);
	static void StopMeasure(const std::string& label);
	static void CleanMeasure(const std::string& label);
	static void PrintMeasure(const std::string& label);

	static void CleanAllMeasures();
	static void PrintAllMeasures();
};