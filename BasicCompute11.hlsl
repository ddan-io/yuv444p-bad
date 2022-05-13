
struct pix { float b; float g; float r; float a; };

StructuredBuffer<pix> tex : register(t0);
RWByteAddressBuffer yuvb : register(u0);

static const float3 cy = float3(0.2126, 0.7152, 0.0722) * 219.0 / 255.0;
static const float3 cu = float3(-0.1146, -0.3854, 0.5) * 224.0 / 255.0;
static const float3 cv = float3(0.5, -0.4542, -0.0458) * 224.0 / 255.0;

[numthreads(16, 1, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
	uint planesz = 1280 * 720;

	uint outY = 0;
	uint outU = 0; 
	uint outV = 0;

	float3 pix1 = float3(tex[4*id.x].r, tex[4*id.x].g, tex[4*id.x].b);
	float3 pix2 = float3(tex[4*id.x + 1].r, tex[4*id.x + 1].g, tex[4*id.x + 1].b);
	float3 pix3 = float3(tex[4*id.x + 2].r, tex[4*id.x + 2].g, tex[4*id.x + 2].b);
	float3 pix4 = float3(tex[4*id.x + 3].r, tex[4*id.x + 3].g, tex[4*id.x + 3].b);

	outY = uint(clamp(16. + dot(cy, pix1), 16., 235.));
	outY |= uint(clamp(16. + dot(cy, pix2), 16., 235.)) << 8;
	outY |= uint(clamp(16. + dot(cy, pix3), 16., 235.)) << 16;
	outY |= uint(clamp(16. + dot(cy, pix4), 16., 235.)) << 24;

	outU = uint(clamp(128.5 + dot(cu, pix1), 16., 240.));
	outU |= uint(clamp(128.5 + dot(cu, pix2), 16., 240.)) << 8;
	outU |= uint(clamp(128.5 + dot(cu, pix3), 16., 240.)) << 16;
	outU |= uint(clamp(128.5 + dot(cu, pix4), 16., 240.)) << 24;
	
	outV = uint(clamp(128.5 + dot(cv, pix1), 16., 240.));
	outV |= uint(clamp(128.5 + dot(cv, pix2), 16., 240.)) << 8;
	outV |= uint(clamp(128.5 + dot(cv, pix3), 16., 240.)) << 16;
	outV |= uint(clamp(128.5 + dot(cv, pix4), 16., 240.)) << 24;

	yuvb.Store(4*id.x, outY);
	yuvb.Store(4*id.x + planesz, outU);
	yuvb.Store(4*id.x + 2*planesz, outV);
}
