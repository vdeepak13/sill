#include <iostream>
#include <string>

#include <prl/parallel/pthread_tools.hpp>
#include <prl/parallel/blocking_queue.hpp>

#include <prl/macros_def.hpp>

using namespace std;

typedef blocking_queue<string> string_queue_type;

class echo_printer {
private:
  string_queue_type* m_queue;
public:
  echo_printer(string_queue_type* queue) : m_queue(queue) { }
  void run() {
    while(true) {
      string s = m_queue->blocking_dequeue();
      cout << "Echo: " << s << endl;
    }
  }
};
  

int main(int argc, char** argv) {
  cout << "Testing blocking queue! " 
       << "Enter some text and press enter " << endl;

  string_queue_type queue;
  echo_printer printer(&queue);
  thread_group<echo_printer> threads;
  threads.launch(&printer);

  while(true) {
    string s;
    cin >> s;
    queue.enqueue(s);
  }


}


