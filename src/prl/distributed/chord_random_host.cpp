#include <iostream>
#include <stdexcept>

#include <prl/distributed/chord_random_host.hpp>

#include <chord_types.h>
#include "chord/chord_prot.h" // for chord_findarg 
#include "chord/misc_utils.h" // for make_chord_node
#include "chord/rpclib.h" // for doRPC
#include <id_utils.h> // for make_randomID

namespace prl {

  namespace {

    // A callback function that receives the result of the findroute query
    void findroute_cb(chord_nodelistres* route, 
		      const chord_random_host* obj,
		      clnt_stat err) {
      using namespace std;
      // extract the last node from the route
      if (err) {
	// typically a time-out (e.g. if Chord is not running locally)
	// cerr << "findroute RPC failed: " << err << "\n";
	// exit(-1);
	obj->failed();
      } else if (route->status == CHORD_RPCFAILURE) {
	obj->failed();
      } else if (route->status != CHORD_OK) {
	cerr << "findroute RPC bad status: " << route->status << "\n";
	exit(-1);
      } else if (route->resok->nlist.size () < 1) {
	cerr << "findroute RPC returned no route!\n";
	exit(-1);
      } else {
	size_t i = route->resok->nlist.size() - 1;
	chord_node result = make_chord_node(route->resok->nlist[i]);
	obj->set_result(result);
      }

      delete route;
      amain_exit = true; // this will cause the amain function to exit
      // this requires a patch to the sfslite-0.8 distribution
      // see svn/projects/libs/sfslite-0.8.tar.gz 
    }
  }

  chord_random_host::chord_random_host(const std::string& ip_address,
				       unsigned short chord_port,
				       unsigned short client_port)
    : contact_node(*new chord_node()),
      client_port(client_port),
      random_node(*new chord_node()) { 
    contact_node.r.hostname = ip_address.c_str();
    contact_node.r.port = chord_port;
    contact_node.x = make_chordID(contact_node.r.hostname, contact_node.r.port);
    contact_node.vnode_num = 0; // what is this?
  }

  chord_random_host::~chord_random_host() {
    delete &contact_node;
    delete &random_node;
  }

  std::pair<std::string, unsigned short> chord_random_host::operator()() const {
    // Perform 5 trials
    for(size_t i = 0; i < 5; i++) {
      chordID x = make_randomID();

      // invoke the findroute RPC and wait for the result
      findroute(x);
      amain();
      if (success) {
	// the result is now in random_node
	return std::make_pair(random_node.r.hostname.cstr(), client_port);
      }
    }

    throw std::runtime_error("RPC failed repeatedly");
  }

  void chord_random_host::findroute(const chordID& key) const {
    ptr<chord_findarg> fa = New refcounted<chord_findarg>();
    fa->x = key;
    fa->return_succs = false; 
    // true = return all the successors starting with fa->x
    chord_nodelistres* route = New chord_nodelistres();
    doRPC(contact_node, chord_program_1, 
	  CHORDPROC_FINDROUTE, fa, route, wrap(&findroute_cb, route, this));
  }

  void chord_random_host::set_result(const chord_node& result) const {
    random_node = result;
    success = true;
  }

  void chord_random_host::failed() const {
    success = false;
  }
  
}

