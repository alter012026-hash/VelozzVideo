import asyncio
from video_factory.render_pipeline import render_script, RenderRequest, RenderScene

async def main():
    req = RenderRequest(
        scenes=[RenderScene(id='1', text='Teste local de narracao')],
        format='9:16',
        scriptTitle='Teste_local'
    )
    try:
        out = await render_script(req)
        print('OK', out)
    except Exception as e:
        import traceback
        traceback.print_exc()

asyncio.run(main())
