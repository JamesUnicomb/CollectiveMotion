#include <math.h>
#define FLOATMAX 1e10
#define EPS2 0.000001


__global__ void update(float3 *pos, 
                       float3 *vel, 
                       float3 *pos_, 
                       float3 *vel_, 
                       float3 *noise, 
                       int n, 
                       float timedelta,
                       float ra = 0.8, 
                       float rb = 0.2, 
                       float re = 0.5, 
                       float r0 = 1.0, 
                       float b = 5.0, 
                       float J = 0.001)
{
    float d[nc] = {[0 ... (nc-1)] = FLOATMAX};
    int ni[nc];

    int id = threadIdx.x + blockDim.x*blockIdx.x;

    for (int j = 0; j < nc; j++)
    {
        for (int sub_id = 0; sub_id < n; sub_id++)
        {
            float3 r;

            r.x = pos_[sub_id].x - pos_[id].x;
            r.y = pos_[sub_id].y - pos_[id].y;
            r.z = pos_[sub_id].z - pos_[id].z;

            float dist = sqrtf(r.x * r.x + r.y * r.y + r.z * r.z);

            if ((dist < d[j]) && (id != sub_id) && (dist > d[j-1]))
            {
                d[j] = dist;
                ni[j] = sub_id;
            }
            else if ((dist < d[0]) && (id != sub_id) && (j == 0))
            {
                d[j] = dist;
                ni[j] = sub_id;
            }
        }
    }

    
    float3 cohesion;
    float3 alignment;

    for (int k = 0; k < nc; k++)
    {
        float f;
        float3 r;

        r.x = pos_[ni[k]].x - pos_[id].x;
        r.y = pos_[ni[k]].y - pos_[id].y;
        r.z = pos_[ni[k]].z - pos_[id].z;

        float dist = sqrtf(r.x * r.x + r.y * r.y + r.z * r.z);

        if (dist < rb) 
        {
            f = -1.0 * FLOATMAX;
        }
        else if ((rb <= dist) && (dist < ra)) 
        {
            f = 0.25 * ((dist - re) / (ra - re));
        }
        else if ((ra <= dist) && (dist < r0)) 
        {
            f = 1.0;
        }
        else if ((r0 <= dist))
        {
            f = 0.0;
        }

        cohesion.x += f * r.x;
        cohesion.y += f * r.y;
        cohesion.z += f * r.z;

        alignment.x += vel_[ni[k]].x;
        alignment.y += vel_[ni[k]].y;
        alignment.z += vel_[ni[k]].z;
    }

    vel[id].x = J * (float)(nc) * alignment.x + b * cohesion.x + (float)(nc) * noise[id].x;
    vel[id].y = J * (float)(nc) * alignment.y + b * cohesion.y + (float)(nc) * noise[id].y;
    vel[id].z = J * (float)(nc) * alignment.z + b * cohesion.z + (float)(nc) * noise[id].z;

    float V = sqrtf(vel[id].x * vel[id].x + vel[id].y * vel[id].y + vel[id].z * vel[id].z + EPS2);

    vel[id].x /= V;
    vel[id].y /= V;
    vel[id].z /= V;

    pos[id].x = pos_[id].x + timedelta * vel[id].x;
    pos[id].y = pos_[id].y + timedelta * vel[id].y;
    pos[id].z = pos_[id].z + timedelta * vel[id].z;
}