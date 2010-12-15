// this file is based on the findroute.C program in the Chord distribution

#include <iostream>
#include <vector>
#include <boost/lexical_cast.hpp>
//#include <boost/thread/thread.hpp>

#include <prl/distributed/address_functions.hpp>
#include <prl/distributed/sfslite_io.hpp> // for str & bigint ostream output
#include <prl/distributed/chord_lookup.hpp>

#include <id_utils.h>

/*
// Chord includes
#include <chord_types.h>
#include <id_utils.h>


#include <prl/distributed/chord/chord_prot.h> // for chord_findarg 
#include <prl/distributed/chord/misc_utils.h>  // for make_randomID
#include <prl/distributed/chord/rpclib.h> // for doRPC

// u_int64_t starttime;

size_t requests_left;

void findroute_cb(//chord_node n,
		  ptr<chord_findarg> fa,
		  chord_nodelistres* route,
		  clnt_stat err) {
  using namespace std;
  if (err) {
    cerr << "findroute RPC failed: " << err << "\n";
  } else if (route->status != CHORD_OK) {
    cerr << "findroute RPC bad status: " << route->status << "\n";
  } else if (route->resok->nlist.size () < 1) {
    cerr << "findroute RPC returned no route!\n";
  } else {
    //cout << " key " << (fa->x>>144) << " rsz " << route->resok->nlist.size ();
    cout << "Found " << fa->x << endl; // << " via:\n";
    //for (size_t i = 0; i < route->resok->nlist.size (); i++) {
    size_t i = route->resok->nlist.size() - 1;
    chord_node z = make_chord_node (route->resok->nlist[i]);
    chordID n    = z.x;
    str host     = z.r.hostname;
    u_short port = z.r.port;
    int index    = z.vnode_num;
    assert (index >= 0);
    cout << i << ": "
    	 << n << " " << host << " " << port << " " << index << "\n";
  }
  delete route;
  requests_left--;
  if (!requests_left)
    amain_exit = true;
  //exit(0);
}

void findroute(chordID key, const chord_node& dst) {
  ptr<chord_findarg> fa = New refcounted<chord_findarg> ();
  fa->x = key;
  fa->return_succs = false;
  chord_nodelistres *route = New chord_nodelistres ();
  //starttime = getusec ();
  doRPC(dst, chord_program_1,
	CHORDPROC_FINDROUTE, fa, route, wrap(&findroute_cb, fa, route));
}

*/

int main (int argc, char *argv[]) {
  using namespace std;
  chordID x;

  if (argc < 4) {
    cerr << "Usage: chord_lookup host port count" << endl;
    return -1;
  }

  /*
  chord_node dst;
  // dst.r.hostname should be an IP address of the host
  dst.r.hostname = prl::address_string(argv[1]).c_str();
  dst.r.port = boost::lexical_cast<int32_t>(argv[2]);
  dst.x = make_chordID(dst.r.hostname, dst.r.port);
  dst.vnode_num = 0;
  */
  
  std::string ip_address = prl::address_string(argv[1]);
  unsigned short port    = boost::lexical_cast<unsigned short>(argv[2]);
  
  prl::chord_lookup chord(ip_address, port);
  size_t count = boost::lexical_cast<size_t>(argv[3]);
  //size_t delay = boost::lexical_cast<size_t>(argv[4]);

//   bool ok = str2chordID (argv[3], x);
//   if (!ok) {
//     cerr << "Invalid key." << endl;
//     return -1;


  //cout << "Host: " << dst.r.hostname << " " << dst.r.port << endl;
  //cout << "Key: " << x << endl;

  //requests_left = count;
  std::vector<chordID> keys;
  for(size_t i = 0; i < count; i++) {
    x = make_randomID();
    chord.lookup(x);
    keys.push_back(x);
    //findroute(x, dst);
  }

  chord.run();
  
  for(size_t i = 0; i < keys.size(); i++) {
    if (chord.contains(keys[i])) {
      cout << "Key " << keys[i] << ": " << chord.result(keys[i]).x << endl;
    } else {
      cout << "Key " << keys[i] << ": missing" << endl;
    }
  }

  //boost::this_thread::sleep(boost::posix_time::seconds(delay));
  
  //amain();

  //  findroute(x, dst);
  // amain();

  // we do not really need to invoke amain on a separate thread
  // but this is try to if it works
  //boost::thread thread(&amain);


  //while(1); // wait for the results
  
}
