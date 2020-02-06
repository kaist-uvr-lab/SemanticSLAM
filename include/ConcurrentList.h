#pragma once
#include <mutex>
#include <list>
//#include <iterator>
//#include <condition_variable>

template <typename T>

class ConcurrentList {
	public:
	typedef typename std::list<T>::iterator iterator;
	iterator begin()
	{
		std::unique_lock<std::mutex> lock(mutex_);
		std::list<T>::iterator iter = list_.begin();
		lock.unlock();
		_cond.notify_one();
		return iter;
	}
	iterator end() {
		std::unique_lock<std::mutex> lock(mutex_);
		std::list<T>::iterator iter = list_.end();
		lock.unlock();
		_cond.notify_one();
		return iter;
	}
	iterator erase(iterator lit) {
		std::unique_lock<std::mutex> lock(mutex_);
		std::list<T>::iterator iter = list_.erase(lit);
		lock.unlock();
		_cond.notify_one();
		return iter;
	}
	void push_back(const T& item) {
		std::unique_lock<std::mutex> lock(mutex_);
		list_.push_back(item);
		lock.unlock();
		_cond.notify_one();
	}
	void clear() {
		std::unique_lock<std::mutex> lock(mutex_);
		list_.clear();
		lock.unlock();
		_cond.notify_one();
	}

private:
	std::list<T> list_;
	std::mutex mutex_;
	std::condition_variable _cond;
};