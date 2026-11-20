import com.cburch.logisim.circuit.Circuit;
import com.cburch.logisim.circuit.CircuitState;
import com.cburch.logisim.circuit.Wire;
import com.cburch.logisim.comp.Component;
import com.cburch.logisim.comp.ComponentDrawContext;
import com.cburch.logisim.comp.EndData;
import com.cburch.logisim.file.Loader;
import com.cburch.logisim.file.LogisimFile;
import com.cburch.logisim.proj.Project;
import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.Collections;
import java.util.Locale;
import javax.imageio.ImageIO;

/**
 * Loads a .circ with the real Logisim-evolution classes and reports what Logisim sees.
 *
 * <pre>
 *   ports  FILE            JSON: every component's factory, location, bounds, and port ends
 *   render FILE CIRC OUT [SCALE]   PNG of one circuit, drawn by Logisim's own painter
 * </pre>
 */
public class LogisimProbe {
  public static void main(String[] args) throws Exception {
    System.setProperty("java.awt.headless", "true");
    if (args.length < 2) {
      System.err.println("usage: LogisimProbe ports FILE | render FILE CIRCUIT OUT.png [SCALE]");
      System.exit(2);
    }
    final var file = new Loader(null).openLogisimFile(new File(args[1]), Collections.emptyMap());
    switch (args[0]) {
      case "ports" -> ports(file);
      case "nets" -> nets(file, args[2]);
      case "render" -> render(file, args[2], args[3], args.length > 4 ? Double.parseDouble(args[4]) : 1.0);
      default -> {
        System.err.println("unknown command " + args[0]);
        System.exit(2);
      }
    }
    System.exit(0);
  }

  private static String q(String s) {
    final var b = new StringBuilder("\"");
    for (final var ch : s.toCharArray()) {
      switch (ch) {
        case '"' -> b.append("\\\"");
        case '\\' -> b.append("\\\\");
        case '\n' -> b.append("\\n");
        default -> {
          if (ch < 0x20) b.append(String.format(Locale.ROOT, "\\u%04x", (int) ch));
          else b.append(ch);
        }
      }
    }
    return b.append('"').toString();
  }

  private static void ports(LogisimFile file) {
    final var out = new StringBuilder("{\"circuits\":[");
    var firstCirc = true;
    for (final Circuit circ : file.getCircuits()) {
      if (!firstCirc) out.append(',');
      firstCirc = false;
      out.append("{\"name\":").append(q(circ.getName())).append(",\"components\":[");
      var firstComp = true;
      for (final Component comp : circ.getNonWires()) {
        if (!firstComp) out.append(',');
        firstComp = false;
        final var loc = comp.getLocation();
        final var b = comp.getBounds();
        out.append("{\"name\":").append(q(comp.getFactory().getName()))
            .append(",\"loc\":[").append(loc.getX()).append(',').append(loc.getY()).append(']')
            .append(",\"bounds\":[").append(b.getX()).append(',').append(b.getY()).append(',')
            .append(b.getWidth()).append(',').append(b.getHeight()).append("],\"ends\":[");
        var firstEnd = true;
        for (final EndData end : comp.getEnds()) {
          if (!firstEnd) out.append(',');
          firstEnd = false;
          final var el = end.getLocation();
          final var type = end.getType() == EndData.INPUT_ONLY ? "in"
              : end.getType() == EndData.OUTPUT_ONLY ? "out" : "inout";
          out.append("{\"loc\":[").append(el.getX()).append(',').append(el.getY())
              .append("],\"width\":").append(end.getWidth().getWidth())
              .append(",\"type\":\"").append(type).append("\"}");
        }
        out.append("]}");
      }
      out.append("],\"wires\":[");
      var firstWire = true;
      for (final Wire w : circ.getWires()) {
        if (!firstWire) out.append(',');
        firstWire = false;
        out.append('[').append(w.getEnd0().getX()).append(',').append(w.getEnd0().getY()).append(',')
            .append(w.getEnd1().getX()).append(',').append(w.getEnd1().getY()).append(']');
      }
      out.append("]}");
    }
    out.append("]}");
    System.out.println(out);
  }

  /** For every component end: the connected wire set Logisim sees there ("-" if no wire). */
  private static void nets(LogisimFile file, String circName) {
    final var circ = file.getCircuit(circName);
    final var out = new StringBuilder("{\"ends\":[");
    var first = true;
    for (final Component comp : circ.getNonWires()) {
      for (final EndData end : comp.getEnds()) {
        final var loc = end.getLocation();
        String key = "-";
        final var here = circ.getWires(loc);
        if (!here.isEmpty()) {
          final var set = circ.getWireSet(here.iterator().next());
          var min = (String) null;
          for (final Wire w : circ.getWires()) {
            if (set.containsWire(w)) {
              final var s = String.format(Locale.ROOT, "%08d,%08d", w.getEnd0().getX() + 100000, w.getEnd0().getY() + 100000);
              if (min == null || s.compareTo(min) < 0) min = s;
            }
          }
          key = min;
        }
        if (!first) out.append(',');
        first = false;
        out.append("{\"comp\":").append(q(comp.getFactory().getName()))
            .append(",\"loc\":[").append(comp.getLocation().getX()).append(',').append(comp.getLocation().getY())
            .append("],\"end\":[").append(loc.getX()).append(',').append(loc.getY())
            .append("],\"net\":").append(q(key)).append('}');
      }
    }
    out.append("]}");
    System.out.println(out);
  }

  private static void render(LogisimFile file, String circName, String outPath, double scale)
      throws Exception {
    final var circ = file.getCircuit(circName);
    if (circ == null) {
      System.err.println("no circuit named " + circName);
      System.exit(2);
    }
    final var proj = new Project(file);
    final var probe = new BufferedImage(1, 1, BufferedImage.TYPE_INT_RGB);
    final var bds = circ.getBounds(probe.getGraphics()).expand(20);
    final var w = Math.max(1, (int) Math.round(bds.getWidth() * scale));
    final var h = Math.max(1, (int) Math.round(bds.getHeight() * scale));
    final var img = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
    final var base = (Graphics2D) img.getGraphics();
    base.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
    base.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_ON);
    base.setColor(Color.white);
    base.fillRect(0, 0, w, h);
    base.setColor(Color.black);
    final var g = (Graphics2D) base.create();
    g.scale(scale, scale);
    g.translate(-bds.getX(), -bds.getY());
    final CircuitState state = proj.getCircuitState(circ);
    final var ctx = new ComponentDrawContext(null, circ, state, base, g, true);
    circ.draw(ctx, null);
    ImageIO.write(img, "PNG", new File(outPath));
  }
}
