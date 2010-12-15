#ifndef _RPCLIB_H_
#define _RPCLIB_H_

#include <arpc.h>

struct chord_node;
extern unsigned int rpclib_timeout;

void doRPC (const chord_node &n, const rpc_program &prog,
	    int procno, const void *in, void *out, aclnt_cb cb);

#endif /* _RPCLIB_H_ */
