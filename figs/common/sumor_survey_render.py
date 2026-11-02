"""VTK helpers for rendering the saved layered-sphere survey.

These functions only render supplied datasets; they do not generate meshes.
"""


import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

from .sumor_paper_figure_style import TEXT_PT, configure


PAPER_WIDTH_CM = 15.92


def _paper_font(prop, width_pixels=2100, size_pt=TEXT_PT):
    """Use the real Times file and physical points, not VTK's pixel units."""
    from matplotlib.font_manager import FontProperties, findfont
    configure()
    font = findfont(FontProperties(family='Times New Roman'), fallback_to_default=False)
    prop.SetFontFamily(vtk.VTK_FONT_FILE)
    prop.SetFontFile(font)
    prop.SetFontSize(round(size_pt * width_pixels / (PAPER_WIDTH_CM / 2.54) / 72))
    prop.SetBold(False)
    prop.SetItalic(False)


COLORS = [(0.75, 0.86, 0.96), (0.91, 0.77, 0.49),
          (0.44, 0.72, 0.82), (0.55, 0.69, 0.45), (0.90, 0.20, 0.15)]


TX_COLOR = (.82, .04, .49)


RX_COLOR = (.02, .27, .85)


def preview(grid, path, topography=False, add_survey=None, show_descriptions=True):
    lookup = vtk.vtkLookupTable()
    lookup.SetNumberOfTableValues(5)
    lookup.SetRange(0, 4)
    lookup.Build()
    for index, color in enumerate(COLORS):
        lookup.SetTableValue(index, *color, 1.)
    mapper = vtk.vtkDataSetMapper()
    mapper.SetInputData(grid)
    mapper.SetScalarModeToUseCellFieldData()
    mapper.SelectColorArray('region_id')
    mapper.SetLookupTable(lookup)
    mapper.SetScalarRange(0, 4)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().EdgeVisibilityOn()
    actor.GetProperty().SetEdgeColor(.22, .26, .29)
    actor.GetProperty().SetLineWidth(.3)
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(1., 1., 1.)
    renderer.AddActor(actor)
    renderer.GetActiveCamera().SetPosition(7000., -11000., 6500.)
    renderer.GetActiveCamera().SetFocalPoint(0., 0., -1300.)
    renderer.GetActiveCamera().SetViewUp(0., 0., 1.)
    renderer.GetActiveCamera().ParallelProjectionOn()
    renderer.ResetCamera()
    if show_descriptions:
        labels = vtk.vtkTextActor()
        labels.SetInput('Layered earth + spherical conductor | tetrahedral mesh\n' +
                        ('Three undulating layer tops; relief decreases with depth\n'
                         if topography else 'Layer 1: 0 to -300 m | Layer 2: -300 to -1000 m\n') +
                        'Sphere: center (0, 0, -650) m, radius 250 m (red)')
        _paper_font(labels.GetTextProperty())
        labels.GetTextProperty().SetColor(.12, .15, .18)
        labels.SetDisplayPosition(25, 25)
        renderer.AddActor2D(labels)
    window = vtk.vtkRenderWindow()
    window.SetOffScreenRendering(1)
    window.SetSize(1500, 1100)
    window.SetMultiSamples(0)
    window.AddRenderer(renderer)
    if add_survey is not None:
        add_survey(renderer)
    window.Render()
    capture = vtk.vtkWindowToImageFilter()
    capture.SetInput(window)
    capture.ReadFrontBufferOff()
    capture.Update()
    writer = vtk.vtkPNGWriter()
    writer.SetFileName(str(path))
    writer.SetInputConnection(capture.GetOutputPort())
    writer.Write()
    window.Finalize()


def _actor(renderer, data, color, opacity=1.):
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(data)
    mapper.ScalarVisibilityOff()
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(*color)
    actor.GetProperty().SetOpacity(opacity)
    renderer.AddActor(actor)
    return actor


def _tube(renderer, data, color, radius):
    tube = vtk.vtkTubeFilter()
    tube.SetInputData(data)
    tube.SetRadius(radius)
    tube.SetNumberOfSides(10)
    tube.CappingOn()
    tube.Update()
    return _actor(renderer, tube.GetOutput(), color)


def _dots(renderer, data, color, radius):
    sphere = vtk.vtkSphereSource()
    sphere.SetRadius(radius)
    sphere.SetThetaResolution(10)
    sphere.SetPhiResolution(10)
    glyph = vtk.vtkGlyph3D()
    glyph.SetInputData(data)
    glyph.SetSourceConnection(sphere.GetOutputPort())
    glyph.ScalingOff()
    glyph.Update()
    return _actor(renderer, glyph.GetOutput(), color)


def _label(renderer, text, position, color=(.12, .15, .18), size=TEXT_PT):
    actor = vtk.vtkBillboardTextActor3D()
    actor.SetInput(text)
    actor.SetPosition(*position)
    prop = actor.GetTextProperty()
    _paper_font(prop, renderer.GetRenderWindow().GetSize()[0], size)
    prop.SetColor(*color)
    prop.SetBold(False)
    prop.SetBackgroundColor(1., 1., 1.)
    prop.SetBackgroundOpacity(.85)
    renderer.AddActor(actor)


def _segments(renderer, segments, color=(.2, .2, .2), width=2.):
    points, lines = vtk.vtkPoints(), vtk.vtkCellArray()
    for start, end in segments:
        first = points.InsertNextPoint(start)
        second = points.InsertNextPoint(end)
        lines.InsertNextCell(2)
        lines.InsertCellPoint(first)
        lines.InsertCellPoint(second)
    poly = vtk.vtkPolyData()
    poly.SetPoints(points)
    poly.SetLines(lines)
    actor = _actor(renderer, poly, color)
    actor.GetProperty().SetLineWidth(width)


def render_survey(cutaway, ground, tx_poly, rx_lines, rx_points,
                  electrodes, projection, output_png, *, show_descriptions=True):
    def decorate(renderer):
        window = renderer.GetRenderWindow()
        window.SetSize(2100, 1540)
        window.SetAlphaBitPlanes(1)
        renderer.SetUseDepthPeeling(1)
        renderer.SetMaximumNumberOfPeels(100)
        renderer.SetOcclusionRatio(.1)
        # The foreground earth is cut away; retain a transparent ground patch
        # so its receivers have a visible surface without hiding the sphere.
        box = vtk.vtkBox()
        box.SetBounds(-2500., 600., -650., 650., -1500., 1500.)
        clip = vtk.vtkClipPolyData()
        clip.SetInputData(ground)
        clip.SetClipFunction(box)
        clip.InsideOutOn()
        clip.Update()
        _actor(renderer, clip.GetOutput(), (.80, .69, .48), .20)
        _tube(renderer, tx_poly, TX_COLOR, 12.)
        _dots(renderer, electrodes, TX_COLOR, 22.)
        _tube(renderer, rx_lines, RX_COLOR, 3.)
        _dots(renderer, rx_points, RX_COLOR, 9.)
        _dots(renderer, projection, (.08, .08, .08), 15.)
        center_z = projection.GetPoint(0)[2]
        tx_z = np.interp(-2000., vtk_to_numpy(tx_poly.GetPoints().GetData())[:, 0],
                         vtk_to_numpy(tx_poly.GetPoints().GetData())[:, 2])
        height = 420.
        ticks = [((-2000., 0., tx_z), (-2000., 0., height+45.)),
                 ((0., 0., center_z), (0., 0., height+45.))]
        dashes = [((x, 0., height), (x+45., 0., height))
                  for x in np.arange(-2000., 0., 75.)]
        _segments(renderer, ticks+dashes)
        _label(renderer, '2 km (horizontal)', (-1500., 0., height+400.))
        _label(renderer, 'Tx: 500 m', (-2500., -150., 160.), TX_COLOR)
        _label(renderer, 'Rx: 10 lines x 20 points\n50 m point / line spacing',
               (800., -520., 510.), RX_COLOR)
        if show_descriptions:
            title = vtk.vtkTextActor()
            title.SetInput('Terrain-following transmitter and surface receivers\n'
                           'Magenta: Tx + electrodes   |   Blue: y-directed Rx lines\n'
                           'Black dot: surface projection of sphere center')
            _paper_font(title.GetTextProperty(), size_pt=TEXT_PT)
            title.GetTextProperty().SetColor(.12, .15, .18)
            title.SetDisplayPosition(35, 1400)
            renderer.AddActor2D(title)
    preview(cutaway, output_png, topography=True, add_survey=decorate,
            show_descriptions=show_descriptions)
