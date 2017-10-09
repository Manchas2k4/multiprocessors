#include <iostream>
#include <tbb/task_scheduler_init.h>
#include <tbb/parallel_while.h>
#include "utils/cppheader.h"
#include "utils/list.h"

using namespace std;
using namespace tbb;

const int SIZE = 100000000;

class NodeStream {
private:
	Node* current;
	
public:
	NodeStream(Node *head) : current(head) {}
	
	bool pop_if_present(Node* &node) {
		if (current) {
			node = current;
			current = current->next;
			return true;
		} else {
			return false;
		}
	}
};

class Apply {
public:
	void operator() (Node* node) const {
		node->value *= 2;
	}
	
	typedef Node* argument_type;
};

void one_thread(Node *head) {
	Node *p = head;
	while (p) {
		p->value *= 2;
		p = p->next;
	}
}

int main(int argc, char* argv[]) {
	Timer t;
	double ms;
	
	List *lst = create_list();
	for (int i = SIZE; i > 0; i--) {
		add_first(lst, i);
	}

	t.start();
	one_thread(lst->head);
	ms = t.stop();
	cout << "one thread time = " << ms << " ms\n";
	
	parallel_while<Apply> w;
	NodeStream stream(lst->head);
	Apply task;
	t.start();
	w.run(stream, task);
	ms = t.stop();
	cout << "tbb time = " << ms << " ms\n";
	
	remove_all(lst);
	free(lst);
	return 0;
}	
	
