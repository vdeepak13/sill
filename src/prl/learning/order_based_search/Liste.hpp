// Liste.hpp
//	Data structures to manage LISP-like lists.  
//
//      As in LISP, a list can contain any type of data structure
//	as an item on the list: thread control blocks, 
//	pending interrupts, etc.  Allocation and deallocation of the
//	items on the list are to be done by the caller.
//
// Copyright (c) 1992-1996 The Regents of the University of California.
// All rights reserved.  See copyright.h for copyright notice and limitation 
// of liability and disclaimer of warranty provisions.

#ifndef PRL_STRUCTURE_SEARCH_ORDER_BASED_LISTE_HPP
#define PRL_STRUCTURE_SEARCH_ORDER_BASED_LISTE_HPP

//#include "copyright.h"
//#include "assert.h"
#include <cassert>
#include <prl/learning/order_based_search/constants.hpp>

// The following class defines a "list element" -- which is
// used to keep track of one item on a list.  It is equivalent to a
// LISP cell, with a "car" ("next") pointing to the next element on the list,
// and a "cdr" ("item") pointing to the item on the list.
//
// This class is private to this module (and classes that inherit
// from this module). Made public for notational convenience.

template <class T>
class ListElement {
  public:
    ListElement(T itm); 	// initialize a list element
    ListElement *next;	     	// next element on list, NULL if this is last
    T item; 	   	     	// item on the list
};

// The following class defines a "list" -- a singly linked list of
// list elements, each of which points to a single item on the list.
// The class has been tested only for primitive types (ints, pointers);
// no guarantees it will work in general.  For instance, all types
// to be inserted into a list must have a "==" operator defined.

template <class T>
class ListIterator;

template <class T>
class List {
  public:
    List();			// initialize the list
    virtual ~List();		// de-allocate the list

    virtual void Prepend(T item);// Put item at the beginning of the list
    virtual void Append(T item); // Put item at the end of the list

    T Front() { return first->item; }
    				// Return first item on list
				// without removing it
    T RemoveFront(); 		// Take item off the front of the list
    void Remove(T item); 	// Remove specific item from list

    bool isInList(T item) const;// is the item in the list?

    unsigned int NumInList() { return numInList;};
    				// how many items in the list?
    bool isEmpty() { return (numInList == 0); };
    				// is the list empty? 

    void Apply(void (*f)(T)) const; 
    				// apply function to all elements in list

    virtual void SanityCheck() const;	
				// has this list been corrupted?
    void SelfTest(T *p, int numEntries);
				// verify module is working

  protected:
    ListElement<T> *first;  	// Head of the list, NULL if list is empty
    ListElement<T> *last;	// Last element of list
    int numInList;		// number of elements in list

friend class ListIterator<T>;
};

// The following class defines a "sorted list" -- a singly linked list of
// list elements, arranged so that "Remove" always returns the smallest 
// element. 
// All types to be inserted onto a sorted list must have a "Compare"
// function defined:
//	   int Compare(T x, T y) 
//		returns -1 if x < y
//		returns 0 if x == y
//		returns 1 if x > y

template <class T>
class SortedList : public List<T> {
  public:
    SortedList(int (*comp)(T x, T y)) : List<T>() { compare = comp;};
    ~SortedList() {};		// base class destructor called automatically

    void Insert(T item); 	// insert an item onto the list in sorted order
    bool isInList(T item) const;// is the item in the list?

    void SanityCheck() const;	// has this list been corrupted?
    void SelfTest(T *p, int numEntries);
				// verify module is working

    using List<T>::isEmpty;

  protected:
    using List<T>::first;
    using List<T>::last;
    using List<T>::numInList;
    using List<T>::RemoveFront;

  private:
    int (*compare)(T x, T y);	// function for sorting list elements

    void Prepend(T item) { Insert(item); }  // *pre*pending has no meaning 
				             //	in a sorted list
    void Append(T item) { Insert(item); }   // neither does *ap*pend 

};

// The following class can be used to step through a list. 
// Example code:
//	ListIterator<T> iter(list); 
//
//	for (; !iter.isDone(); iter.Next()) {
//	    Operation on iter.Item()
//      }

template <class T>
class ListIterator {
  public:
    ListIterator(List<T> *list) { current = list->first; } 
				// initialize an iterator

    bool isDone() { return current == NULL; };
				// return TRUE if we are at the end of the list

    T Item() { ASSERT(!isDone()); return current->item; };
				// return current element on list

    void Next() {
		current = current->next; 
	};		
				// update iterator to point to next
	void ReStart(List<T> *list) { current = list->first;};
  private:
    ListElement<T> *current;	// where we are in the list
};


//----------------------------------------------------------------------
// ListElement<T>::ListElement
// 	Initialize a list element, so it can be added somewhere on a list.
//
//	"itm" is the thing to be put on the list.
//----------------------------------------------------------------------

template <class T>
ListElement<T>::ListElement(T itm)
{
     item = itm;
     next = NULL;	// always initialize to something!
}


//----------------------------------------------------------------------
// List<T>::List
//	Initialize a list, empty to start with.
//	Elements can now be added to the list.
//----------------------------------------------------------------------

template <class T>
List<T>::List()
{ 
    first = last = NULL; 
    numInList = 0;
}

//----------------------------------------------------------------------
// List<T>::~List
//	Prepare a list for deallocation.  
//----------------------------------------------------------------------

template <class T>
List<T>::~List()
{ 
    //ASSERT(isEmpty());		// make sure list is empty
    while(!isEmpty())
    	RemoveFront();
}

//----------------------------------------------------------------------
// List<T>::Append
//      Append an "item" to the end of the list.
//      
//	Allocate a ListElement to keep track of the item.
//      If the list is empty, then this will be the only element.
//	Otherwise, put it at the end.
//
//	"item" is the thing to put on the list.
//----------------------------------------------------------------------

template <class T>
void
List<T>::Append(T item)
{
    ListElement<T> *element = new ListElement<T>(item);

    ASSERT(!isInList(item));
    if (isEmpty()) {		// list is empty
	first = element;
	last = element;
    } else {			// else put it after last
	last->next = element;
	last = element;
    }
    numInList++;
    ASSERT(isInList(item));
}

//----------------------------------------------------------------------
// List<T>::Prepend
//	Same as Append, only put "item" on the front.
//----------------------------------------------------------------------

template <class T>
void
List<T>::Prepend(T item)
{
    ListElement<T> *element = new ListElement<T>(item);

    ASSERT(!isInList(item));
    if (isEmpty()) {		// list is empty
	first = element;
	last = element;
    } else {			// else put it before first
	element->next = first;
	first = element;
    }
    numInList++;
    ASSERT(isInList(item));
}

//----------------------------------------------------------------------
// List<T>::RemoveFront
//      Remove the first "item" from the front of the list.
//	List must not be empty.
// 
// Returns:
//	The removed item.
//----------------------------------------------------------------------

template <class T>
T
List<T>::RemoveFront()
{
    ListElement<T> *element = first;
    T thing;

    ASSERT(!isEmpty());

    thing = first->item;
    if (first == last) {	// list had one item, now has none 
        first = NULL;
	last = NULL;
    } else {
        first = element->next;
    }
    numInList--;
    delete element;
    return thing;
}

//----------------------------------------------------------------------
// List<T>::Remove
//      Remove a specific item from the list.  Must be in the list!
//----------------------------------------------------------------------

template <class T>
void
List<T>::Remove(T item)
{
    ListElement<T> *prev, *ptr;
    T removed;

    ASSERT(isInList(item));

    // if first item on list is match, then remove from front
    if (item == first->item) {	
        removed = RemoveFront();
        ASSERT(item == removed);
    } else {
	prev = first;
        for (ptr = first->next; ptr != NULL; prev = ptr, ptr = ptr->next) {
            if (item == ptr->item) {
		prev->next = ptr->next;
		if (prev->next == NULL) {
		    last = prev;
		}
		delete ptr;
		numInList--;
		break;
	    }
        }
	ASSERT(ptr != NULL);	// should always find item!
    }
   ASSERT(!isInList(item));
}

//----------------------------------------------------------------------
// List<T>::isInList
//      Return TRUE if the item is in the list.
//----------------------------------------------------------------------

template <class T>
bool
List<T>::isInList(T item) const
{ 
    ListElement<T> *ptr;

    for (ptr = first; ptr != NULL; ptr = ptr->next) {
        if (item == ptr->item) {
            return TRUE;
        }
    }
    return FALSE;
}


//----------------------------------------------------------------------
// List<T>::Apply
//      Apply function to every item on a list.
//
//	"func" -- the function to apply
//----------------------------------------------------------------------

template <class T>
void
List<T>::Apply(void (*func)(T)) const
{ 
    ListElement<T> *ptr;

    for (ptr = first; ptr != NULL; ptr = ptr->next) {
        (*func)(ptr->item);
    }
}

//----------------------------------------------------------------------
// SortedList<T>::isInList
//      Return TRUE if the item is in the list.
//----------------------------------------------------------------------

template <class T>
bool
SortedList<T>::isInList(T item) const
{ 
    ListElement<T> *ptr;

    for (ptr = first; ptr != NULL && compare(item, first->item)>=0 ; ptr = ptr->next) {
        if (item == ptr->item) {
            return TRUE;
        }
    }
    return FALSE;
}


//----------------------------------------------------------------------
// SortedList::Insert
//      Insert an "item" into a list, so that the list elements are
//	sorted in increasing order.
//      
//	Allocate a ListElement to keep track of the item.
//      If the list is empty, then this will be the only element.
//	Otherwise, walk through the list, one element at a time,
//	to find where the new item should be placed.
//
//	"item" is the thing to put on the list. 
//----------------------------------------------------------------------

template <class T>
void
SortedList<T>::Insert(T item)
{
    ListElement<T> *element = new ListElement<T>(item);
    ListElement<T> *ptr;		// keep track

    ASSERT(!isInList(item));
    if (isEmpty()) {			// if list is empty, put at front
        first = element;
        last = element;
    } else if (compare(item, first->item) < 0) {  // item goes at front 
	element->next = first;
	first = element;
    } else {		// look for first elt in list bigger than item
        for (ptr = first; ptr->next != NULL; ptr = ptr->next) {
            if (compare(item, ptr->next->item) < 0) {
				element->next = ptr->next;
				ptr->next = element;
				numInList++;
		return;
	    }
	}
	last->next = element;		// item goes at end of list
	last = element;
    }
    numInList++;
    ASSERT(isInList(item));
}

//----------------------------------------------------------------------
// List::SanityCheck
//      Test whether this is still a legal list.
//
//	Tests: do I get to last starting from first?
//	       does the list have the right # of elements?
//----------------------------------------------------------------------

template <class T>
void 
List<T>::SanityCheck() const
{
    ListElement<T> *ptr;
    int numFound;

    if (first == NULL) {
	ASSERT((numInList == 0) && (last == NULL));
    } else if (first == last) {
	ASSERT((numInList == 1) && (last->next == NULL));
    } else {
        for (numFound = 1, ptr = first; ptr != last; ptr = ptr->next) {
	    numFound++;
            ASSERT(numFound <= numInList);	// prevent infinite loop
        }
        ASSERT(numFound == numInList);
        ASSERT(last->next == NULL);
    }
}

//----------------------------------------------------------------------
// List::SelfTest
//      Test whether this module is working.
//----------------------------------------------------------------------

template <class T>
void 
List<T>::SelfTest(T *p, int numEntries)
{
    int i;
    ListIterator<T> *iterator = new ListIterator<T>(this);

    SanityCheck();
    // check various ways that list is empty
    ASSERT(isEmpty() && (first == NULL));
    for (; !iterator->isDone(); iterator->Next()) {
	ASSERT(false);	// nothing on list; should not be reached
    }

    for (i = 0; i < numEntries; i++) {
	 Append(p[i]);
	 ASSERT(isInList(p[i]));
	 ASSERT(!isEmpty());
     }
     SanityCheck();

     // should be able to get out everything we put in
     for (i = 0; i < numEntries; i++) {
	 Remove(p[i]);
         ASSERT(!isInList(p[i]));
     }
     ASSERT(isEmpty());
     SanityCheck();
     delete iterator;
}

//----------------------------------------------------------------------
// SortedList::SanityCheck
//      Test whether this is still a legal sorted list.
//
//	Test: is the list sorted?
//----------------------------------------------------------------------

template <class T>
void 
SortedList<T>::SanityCheck() const
{
    ListElement<T> *prev, *ptr;

    List<T>::SanityCheck();
    if (first != last) {
        for (prev = first, ptr = first->next; ptr != NULL; 
						prev = ptr, ptr = ptr->next) {
            ASSERT(compare(prev->item, ptr->item) <= 0);
        }
    }
}

//----------------------------------------------------------------------
// SortedList::SelfTest
//      Test whether this module is working.
//----------------------------------------------------------------------

template <class T>
void 
SortedList<T>::SelfTest(T *p, int numEntries)
{
    int i;
    T *q = new T[numEntries];

    List<T>::SelfTest(p, numEntries);

    for (i = 0; i < numEntries; i++) {
	 Insert(p[i]);
	 ASSERT(isInList(p[i]));
     }
     SanityCheck();

     // should be able to get out everything we put in
     for (i = 0; i < numEntries; i++) {
	 q[i] = RemoveFront();
         ASSERT(!isInList(q[i]));
     }
     ASSERT(isEmpty());

     // make sure everything came out in the right order
     for (i = 0; i < (numEntries - 1); i++) {
	 ASSERT(compare(q[i], q[i + 1]) <= 0);
     }
     SanityCheck();

     delete q;
};

#endif // PRL_STRUCTURE_SEARCH_ORDER_BASED_LISTE_HPP

