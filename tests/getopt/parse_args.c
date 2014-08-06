#include "parse_args.h"

void print_usage(){
    printf("Usage: ./main.out [OPTIONS] config_file \n");
}

static struct option long_options[] = {
        { "help",      no_argument,       0,  'h' },
        { 0,           0,                 0,   0  }
    };
static char short_options[] = "h";

void parse_args( int argc, char *argv[], char *config_file[] )
{
    int opt = 0;
    int long_opt_index = 0;	

    while( 1 ) {
	opt = getopt_long( argc, argv, short_options, 
			   long_options, &long_opt_index );
	if ( opt == -1 ){
	    break;
	}
        switch( opt ) {
	case 'h' : 
	    print_usage();
	    exit( EXIT_FAILURE );
	    break;
	case '?' :
	    print_usage();
	    exit( EXIT_FAILURE );
	default: 
	    print_usage(); 
	    exit( EXIT_FAILURE );
        }
    }
    
    if ( ( optind == argc ) || ( argc - optind > 1 )  ){
	printf( "Error: expecting a single config file. \n" );
	print_usage(); 
	exit( EXIT_FAILURE );
    }

    *config_file = argv[ optind ];

    return;
}