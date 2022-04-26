#include "TimeLogger.h"
#include <iostream>

TimeLogger* TimeLogger::_instance = nullptr;
std::unordered_map<std::string, TimeLogger::TimeDiff_t> TimeLogger::_timeMeasure;

TimeLogger::TimeLogger() { }

TimeLogger* TimeLogger::getInstance() {
	if (_instance == nullptr)
		return new TimeLogger();
	return _instance;
}

void TimeLogger::StartMeasure(const std::string& label) {
	_timeMeasure[label].first = std::chrono::system_clock::now();
}

void TimeLogger::StopMeasure(const std::string& label) {
	_timeMeasure[label].second = std::chrono::system_clock::now();
}

void TimeLogger::CleanMeasure(const std::string& label) {
	_timeMeasure.erase(label);
}

void TimeLogger::CleanAllMeasures() {
	_timeMeasure.clear();
}

void TimeLogger::PrintAllMeasures()
{
	for (auto& [label, time_measure] : _timeMeasure) {
		std::chrono::duration duration = std::chrono::duration_cast<std::chrono::milliseconds>(time_measure.second - time_measure.first);
		std::cout << "Time measure [" << label << "]: " << duration.count() << std::endl;
	}
}

void TimeLogger::PrintMeasure(const std::string& label) {
	if (_timeMeasure.find(label) == _timeMeasure.end()) {
		std::cerr << "Time measure [" << label << "]: not found!" << std::endl;
		return;
	}

	std::chrono::duration duration = _timeMeasure[label].second - _timeMeasure[label].first;
	std::cout << "Time measure [" << label << "]: " << duration.count() << std::endl;
}
