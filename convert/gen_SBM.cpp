#include <vector>
#include <stdio.h>
#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <functional>
#include <time.h>
#include <vector>
#include <string.h>

using namespace std;

bool cmp(const int& a,const int&b){
	return a<b;
}

//Check parameters
long check_inc(long i, long max) {
    if (i == max) {
        //usage();
        cout<<"i==max"<<endl;
	exit(1);
    }
    return i + 1;
}

void gen_snap(uint vert, int cluster,int mean_degree, int in_degree, int out_degree, uint changeNum, int snap, vector<vector<uint>>& Adj, vector<vector<uint>>& out_Adj,int* clusterID){
    stringstream dy_out;
    dy_out<<"../data/SBM-"<<vert<<"-"<<cluster<<"-"<<in_degree<<"+"<<out_degree<<"_Edgeupdate_snap"<<snap<<".txt";
    cout<<dy_out.str()<<endl;
    
    ofstream fdy;
    fdy.open(dy_out.str());
    if(!fdy){
        cout<<"ERROR:can not open out file"<<endl;
        return;
    }
    //dynamic change
    vector<uint> ChangeNodes;
    for(uint i=0; i<changeNum; i++){
        uint change_node=rand()%vert;
        while( find(ChangeNodes.begin(),ChangeNodes.end(),change_node)!=ChangeNodes.end() ){
            change_node=rand()%vert;
        }
        ChangeNodes.push_back(change_node);
        cout<<"change_node: "<< change_node << "; ori-com: "<<clusterID[change_node]<<endl;
        
        int new_comm = rand()%cluster;
        while(new_comm == clusterID[change_node]){
            new_comm = rand()%cluster;
        }
        clusterID[change_node] = new_comm;
        cout<<"change to com: "<<clusterID[change_node]<<endl;
        
        int new_in_degree = in_degree;
        int new_out_degree = out_degree;
        int drop_degree = Adj[change_node].size();
        cout<<"drop_degree:"<<drop_degree<<endl;
        for(int j=0; j<drop_degree; j++){
            int tmp_index=rand()%Adj[change_node].size();
            uint tmp_node=Adj[change_node][tmp_index];
            Adj[change_node].erase(Adj[change_node].begin() + tmp_index);
            vector<uint>::iterator itr;
            itr=find(Adj[tmp_node].begin(),Adj[tmp_node].end(), change_node);
            int idx=distance(Adj[tmp_node].begin(), itr);
            Adj[tmp_node].erase(itr);
            
            fdy<<change_node<<" "<<tmp_node<<"\n";
            fdy<<tmp_node<<" "<<change_node<<"\n";
        }
        int add_degree = new_in_degree;
        
        int rd_seed = rand()%10;
        if(rd_seed<=6)
            add_degree = mean_degree;
        else
            add_degree = mean_degree + 1;
        
        int dd=0; // find edges in out_adj but have some label with change_node's new_label and delete
        for(int j=0; j<out_Adj[change_node].size(); j++){
            int old_out_neibor = out_Adj[change_node][j];
            if(clusterID[old_out_neibor] == clusterID[change_node]){
                dd += 1;
                //delete 
                vector<uint>::iterator itr;
                itr=find(out_Adj[change_node].begin(),out_Adj[change_node].end(), old_out_neibor);
                out_Adj[change_node].erase(itr);
                itr=find(out_Adj[old_out_neibor].begin(),out_Adj[old_out_neibor].end(), change_node);
                out_Adj[old_out_neibor].erase(itr);
            }
        }
        if(dd>0){
            for(int j=0; j<dd; j++){
                uint tmp_node=rand()%vert;
                while(clusterID[tmp_node]==clusterID[change_node] || find(out_Adj[change_node].begin(),out_Adj[change_node].end(),tmp_node)!=out_Adj[change_node].end() ||tmp_node==change_node){
                    tmp_node=rand()%vert;
                }
                //cout<<"tmp_node="<<tmp_node<<endl;
                if( find(out_Adj[tmp_node].begin(),out_Adj[tmp_node].end(),change_node)==out_Adj[tmp_node].end() ){
                    out_Adj[change_node].push_back(tmp_node);
                    out_Adj[tmp_node].push_back(change_node);
                }else{
                    cout<< "!!!!!!!!!!!!!Error----dd>0/for" <<endl;
                }
                fdy<<change_node<<" "<<tmp_node<<"\n";
                fdy<<tmp_node<<" "<<change_node<<"\n";
            }
        }
        cout<<"add_degree:"<<add_degree<<endl;
        for(int j=0; j<add_degree; j++){
            uint tmp_node=rand()%vert;
            while(clusterID[tmp_node]!=clusterID[change_node] || tmp_node==change_node || find(Adj[change_node].begin(),Adj[change_node].end(),tmp_node)!=Adj[change_node].end()){
                tmp_node=rand()%vert;
            }
            
            Adj[change_node].push_back(tmp_node);
            fdy<<change_node<<" "<<tmp_node<<"\n";
            if( find(out_Adj[tmp_node].begin(),out_Adj[tmp_node].end(),change_node)==out_Adj[tmp_node].end() && find(Adj[tmp_node].begin(),Adj[tmp_node].end(),change_node)==Adj[tmp_node].end() ){
                Adj[tmp_node].push_back(change_node);
                fdy<<tmp_node<<" "<< change_node <<"\n";
            }else{
                cout<<"!!!!!!!!!!!Error----edge is already exist"<<endl;
                return;
            }
        }
    }
    fdy.close();
    stringstream label_out2;
    label_out2<<"../data/SBM-"<<vert<<"-"<<cluster<<"-"<<in_degree<<"+"<<out_degree<<"_label_snap"<<snap<<".txt";
    cout<<label_out2.str()<<endl;
    ofstream f2;
    f2.open(label_out2.str());
    for(uint i=0;i<vert;i++){
        f2<<clusterID[i]<<"\n";
    }
    f2.close();
    
    stringstream new_out;
    new_out<<"../data/SBM-"<<vert<<"-"<<cluster<<"-"<<in_degree<<"+"<<out_degree<<"_snap"<<snap<<".txt";
    cout<<new_out.str()<<endl;

    ofstream fsnap;
    fsnap.open(new_out.str());
    if(!fsnap){
        cout<<"ERROR:can not open out file"<<endl;
        return;
    }
    //save edge list
    for(uint j=0; j<vert; j++){  
        sort(Adj[j].begin(),Adj[j].end(),cmp);
        for(int k=0; k<Adj[j].size(); k++){
            fsnap<<j<<" "<<Adj[j][k]<<"\n";
        }
        sort(out_Adj[j].begin(),out_Adj[j].end(),cmp);
        for(int k=0; k<out_Adj[j].size(); k++){
            fsnap<<j<<" "<<out_Adj[j][k]<<"\n";
        }
    }
    fsnap.close();
}

int main(int argc,char **argv){
    //srand(time(NULL));
    srand(20);
    char *endptr;
    uint vert=10000;
    int cluster=2;
    double in_com=1;
    double between_com=1;
    int in_degree = 20;
    int out_degree = 5;
    uint changeNum = 5;
    int snapeNum=0; 
    int i=1;
    while (i < argc) {
        if (!strcmp(argv[i], "-n")) {
            i = check_inc(i, argc);
            vert = strtod(argv[i],&endptr);
        } else if (!strcmp(argv[i], "-c")) {
            i = check_inc(i, argc);
            cluster = strtod(argv[i],&endptr);
        } else if (!strcmp(argv[i], "-ind")) {
            i = check_inc(i, argc);
            in_degree = strtod(argv[i],&endptr);
        } else if (!strcmp(argv[i], "-outd")) {
            i = check_inc(i, argc);
            out_degree = strtod(argv[i], &endptr);
        } else if (!strcmp(argv[i], "-inp")) {
            i = check_inc(i, argc);
            in_com = strtod(argv[i],&endptr);
        } else if (!strcmp(argv[i], "-outp")) {
            i = check_inc(i, argc);
            between_com = strtod(argv[i], &endptr);
        }else if (!strcmp(argv[i], "-change")) {
            i = check_inc(i, argc);
            changeNum = strtod(argv[i], &endptr);
        }else if (!strcmp(argv[i], "-snap")) {
            i = check_inc(i, argc);
            snapeNum = strtod(argv[i], &endptr);
        } else {
            cout<<"ERROR parameter!!!"<<endl;
            exit(1);
        }
        i++;
    }

    uint N_perCluster=vert/cluster;
    if(in_com<1 && between_com<1){
        in_degree = in_com*N_perCluster;
        out_degree = between_com*N_perCluster;
    }
    
    cout<<"vert="<<vert<<endl;
    cout<<"cluster="<<cluster<<endl;
    cout<<"in_com="<<in_com<<endl;
    cout<<"between_com="<<between_com<<endl;
    cout<<"in_degree="<<in_degree<<endl;
    cout<<"out_degree="<<out_degree<<endl;
    cout<<"N_perCluster="<<N_perCluster<<endl;
    cout<<"snapNum="<<snapeNum<<endl;

    int *clusterID=new int[vert];
    for(uint i=0;i<vert;i++){
        int clusterFlag=i/N_perCluster;
        if(clusterFlag>=cluster){
            clusterFlag=cluster-1;
        }
        clusterID[i]=clusterFlag;
    }

    stringstream label_out;
    label_out<<"../data/SBM-"<<vert<<"-"<<cluster<<"-"<<in_degree<<"+"<<out_degree<<"_label.txt";
    cout<<label_out.str()<<endl;
    ofstream f1;
    f1.open(label_out.str());
    for(uint i=0;i<vert;i++){
        f1<<clusterID[i]<<"\n";
    }
    f1.close();

    stringstream ss_out;
    ss_out<<"../data/SBM-"<<vert<<"-"<<cluster<<"-"<<in_degree<<"+"<<out_degree<<"_init.txt";
    cout<<ss_out.str()<<endl;

    ofstream fout;
    fout.open(ss_out.str());
    if(!fout){
        cout<<"ERROR:can not open out file"<<endl;
        return 0;
    }
    vector<vector<uint>> Adj;
    vector<vector<uint>> out_Adj;
    vector<uint> random_w = vector<uint>(vert);

    for (uint i = 0; i < vert; i++)
    {
        vector<uint> templst;
        Adj.push_back(templst);
        out_Adj.push_back(templst);
        random_w[i] = i;
    }
    random_shuffle(random_w.begin(),random_w.end());
    
    for(uint i=0;i<vert;i++){
        uint w = random_w[i];
        int di = Adj[w].size();
        for(int j=0;j<(in_degree - di);j++){
            uint tmp_node=rand()%N_perCluster;
            tmp_node+=clusterID[w]*N_perCluster;
            while(find(Adj[w].begin(),Adj[w].end(),tmp_node)!=Adj[w].end() || tmp_node==w )
            {
                tmp_node=rand()%N_perCluster;
                tmp_node+=clusterID[w]*N_perCluster;
            }

            Adj[w].push_back(tmp_node);
            if( find(Adj[tmp_node].begin(),Adj[tmp_node].end(),w)==Adj[tmp_node].end() ){
                Adj[tmp_node].push_back(w);
            }

        }
        for(int j=0;j<out_degree;j++){
            uint tmp_node=rand()%vert;
            while(clusterID[tmp_node]==clusterID[w]){
                tmp_node=rand()%vert;
            }
            if(find(out_Adj[w].begin(),out_Adj[w].end(),tmp_node)==out_Adj[w].end() && tmp_node!=w ){
                out_Adj[w].push_back(tmp_node);
                if( find(out_Adj[tmp_node].begin(),out_Adj[tmp_node].end(),w)==out_Adj[tmp_node].end() ){
                    out_Adj[tmp_node].push_back(w);
                }
            }
        }

    }
    int edges = 0;
    for(uint j=0; j<vert; j++){  //init
        sort(Adj[j].begin(),Adj[j].end(),cmp);
        for(int k=0; k<Adj[j].size(); k++){
            fout<<j<<" "<<Adj[j][k]<<"\n";
            edges += 1;
        }
        sort(out_Adj[j].begin(),out_Adj[j].end(),cmp);
        for(int k=0; k<out_Adj[j].size(); k++){
            fout<<j<<" "<<out_Adj[j][k]<<"\n";
        }
    }
    fout.close();
    
    int mean_degree = edges / vert;
    cout << "m=" << edges << ", mean_degree=" << mean_degree << endl;
    for(int i=0; i<snapeNum;i++){
        gen_snap(vert,cluster,mean_degree,in_degree,out_degree,changeNum,i,Adj,out_Adj,clusterID);
    }
    
    delete[] clusterID;
    return 0;
}



